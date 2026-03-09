"""
Spark Batch Processor
Handles large-scale batch inference jobs using PySpark.
Use this when processing thousands of records (e.g. nightly batch scoring).

For real-time single requests: use the Flask API directly.
For medium batches (<100): use the MPI dispatcher.
For large batches (1000+): use this Spark job.

Run: spark-submit spark_job.py --input data/texts.json --output data/results/
"""

import json
import argparse
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

# Try importing PySpark — gracefully skip if not installed
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import udf, col
    from pyspark.sql.types import StructType, StructField, StringType, FloatType
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    logger.warning("PySpark not installed. Install with: pip install pyspark")


def create_spark_session(app_name="DistributedInferenceEngine"):
    """Create a local Spark session (swap master URL for cluster deployment)."""
    return (SparkSession.builder
            .appName(app_name)
            .master("local[*]")  # use all cores locally; set to spark://host:7077 for cluster
            .config("spark.executor.memory", "2g")
            .config("spark.driver.memory", "2g")
            .getOrCreate())


def run_batch_inference(input_path: str, output_path: str):
    """
    Spark job: read texts from JSON, run inference on each partition, write results.

    Spark distributes partitions across executors automatically.
    Each executor runs predict() independently — no shared state.
    """
    if not SPARK_AVAILABLE:
        logger.error("PySpark required. Run: pip install pyspark")
        return

    # Lazy import to avoid loading at module level
    from model import load_model, predict as ml_predict

    spark = create_spark_session()
    logger.info("Spark session started. Reading from %s", input_path)

    # Read input
    df = spark.read.json(input_path)

    # Define inference UDF (runs on each executor independently)
    def infer_udf(text):
        try:
            model, vectorizer = load_model()
            result = ml_predict(model, vectorizer, text)
            return result["label"]
        except Exception:
            return "error"

    sentiment_udf = udf(infer_udf, StringType())

    # Apply inference across all partitions in parallel
    results_df = df.withColumn("label", sentiment_udf(col("text")))

    # Write results
    results_df.write.mode("overwrite").json(output_path)
    logger.info("Batch complete. Results written to %s", output_path)

    count = results_df.count()
    logger.info("Processed %d records.", count)
    spark.stop()
    return count


def simulate_spark_job(texts: list) -> list:
    """
    Simulates Spark partitioning logic without requiring a running cluster.
    Used for local testing and demo purposes.
    """
    from model import load_model, predict

    model, vectorizer = load_model()
    n_partitions = 4
    partitions = [texts[i::n_partitions] for i in range(n_partitions)]

    results = []
    for partition_id, partition in enumerate(partitions):
        for text in partition:
            result = predict(model, vectorizer, text)
            result["partition"] = partition_id
            result["executor"] = f"executor-{partition_id}"
            results.append(result)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spark Batch Inference Job")
    parser.add_argument("--input",  required=True, help="Path to input JSON file")
    parser.add_argument("--output", required=True, help="Path to output directory")
    args = parser.parse_args()
    run_batch_inference(args.input, args.output)
