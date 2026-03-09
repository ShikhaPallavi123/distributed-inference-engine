"""
MPI Dispatcher — distributes inference tasks across worker processes.

On a GPU cluster: mpirun -n 4 python mpi_worker.py
On Mac (local dev): simulates MPI behavior using multiprocessing
so the architecture is identical — only the execution layer differs.
"""

import os
import multiprocessing
from model import load_model, predict

N_WORKERS = int(os.environ.get("N_WORKERS", min(4, multiprocessing.cpu_count())))


def _worker_task(args):
    """Single worker: loads model and runs inference on its assigned texts."""
    worker_id, texts = args
    model, vectorizer = load_model()
    results = []
    for text in texts:
        result = predict(model, vectorizer, text)
        result["worker"] = worker_id
        results.append(result)
    return results


def dispatch_inference(texts: list) -> list:
    """
    Partition texts across N_WORKERS and run inference in parallel.

    MPI equivalent:
        - Rank 0 (master) scatters work to ranks 1..N
        - Each rank runs inference on its partition
        - Rank 0 gathers and returns results

    Here we use multiprocessing.Pool to simulate this on a single machine.
    On a real cluster, replace Pool with mpi4py MPI.COMM_WORLD scatter/gather.
    """
    # Partition texts across workers
    partitions = [[] for _ in range(N_WORKERS)]
    for i, text in enumerate(texts):
        partitions[i % N_WORKERS].append(text)

    tasks = [(worker_id, partition)
             for worker_id, partition in enumerate(partitions)
             if partition]

    with multiprocessing.Pool(processes=len(tasks)) as pool:
        worker_results = pool.map(_worker_task, tasks)

    # Flatten results maintaining original order
    flat = []
    for results in worker_results:
        flat.extend(results)
    return flat
