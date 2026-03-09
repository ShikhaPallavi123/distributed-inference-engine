"""
Distributed AI Inference Engine — Flask API Gateway
Receives inference requests, distributes work to MPI workers,
streams results back to the React frontend via Server-Sent Events.
"""

import os
import json
import time
import logging
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
from mpi_dispatcher import dispatch_inference
from model import load_model, predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load model once at startup
model, vectorizer = load_model()
logger.info("Model loaded and ready.")


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "service": "inference-api"})


@app.route("/api/infer", methods=["POST"])
def infer():
    """
    Single inference request.
    Body: { "text": "some input text" }
    Returns: { "label": "positive", "confidence": 0.87, "worker": 0, "latency_ms": 12 }
    """
    data = request.get_json()
    if not data or not data.get("text"):
        return jsonify({"error": "text field is required"}), 400

    text = data["text"].strip()
    if len(text) > 2000:
        return jsonify({"error": "text too long, max 2000 characters"}), 400

    start = time.time()
    result = predict(model, vectorizer, text)
    result["latency_ms"] = round((time.time() - start) * 1000, 2)
    result["worker"] = 0  # single node
    logger.info("Inferred '%s...' → %s (%.2fms)", text[:40], result["label"], result["latency_ms"])
    return jsonify(result)


@app.route("/api/infer/batch", methods=["POST"])
def infer_batch():
    """
    Batch inference — distributes texts across MPI worker processes.
    Body: { "texts": ["text1", "text2", ...] }
    Returns ranked results from all workers.
    """
    data = request.get_json()
    texts = data.get("texts", [])

    if not texts:
        return jsonify({"error": "texts list cannot be empty"}), 400
    if len(texts) > 100:
        return jsonify({"error": "max 100 texts per batch"}), 400

    start = time.time()
    results = dispatch_inference(texts)
    total_ms = round((time.time() - start) * 1000, 2)

    return jsonify({
        "results": results,
        "total": len(results),
        "total_latency_ms": total_ms,
        "workers_used": len(set(r["worker"] for r in results))
    })


@app.route("/api/infer/stream", methods=["POST"])
def infer_stream():
    """
    Streaming inference — results are sent back one by one via Server-Sent Events
    as each worker finishes. React frontend renders results in real time.
    """
    data = request.get_json()
    texts = data.get("texts", [])

    if not texts:
        return Response("data: {\"error\": \"texts required\"}\n\n",
                        mimetype="text/event-stream")

    def generate():
        for i, text in enumerate(texts[:50]):
            result = predict(model, vectorizer, text)
            result["index"] = i
            result["worker"] = i % 4  # simulate 4 workers
            yield f"data: {json.dumps(result)}\n\n"
            time.sleep(0.05)  # simulate distributed processing delay
        yield "data: {\"done\": true}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
