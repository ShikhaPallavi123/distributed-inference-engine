# Distributed AI Inference Engine

This is my most ambitious project yet. I wanted to build something that forces every layer of the stack to talk to each other — not just a Flask app that calls a model, but a system where work actually distributes across processes, results stream back in real time, and the architecture is designed to scale to real hardware.

The core idea: a React frontend sends text to a Flask API, which dispatches the work across parallel workers using MPI-style partitioning, runs inference on each partition, then streams results back to the UI via Server-Sent Events. On a GPU server you'd swap in the CUDA module for the heavy matrix ops. On Mac I run it all on CPU — same architecture, different backend.

## Architecture

```
React (port 3000)
    │
    ▼  HTTP / SSE
Flask API Gateway (port 5000)
    │
    ├── /api/infer        → single inference (direct)
    ├── /api/infer/batch  → MPI dispatcher → worker pool
    └── /api/infer/stream → SSE stream, one result per worker
            │
            ├── Worker 0: partition[0::4]
            ├── Worker 1: partition[1::4]
            ├── Worker 2: partition[2::4]
            └── Worker 3: partition[3::4]
                    │
                    └── TF-IDF + Logistic Regression
                        (CUDA module for GPU deployment)
```

**Spark** handles large-scale batch jobs separately — when you have thousands of records, you run `spark-submit spark_job.py` and Spark handles the partitioning across executors automatically.

## What each component does

**`backend/api/app.py`** — Flask API gateway. Three endpoints: single inference, batch via MPI, and streaming via SSE. CORS enabled for the React frontend.

**`backend/api/model.py`** — TF-IDF + Logistic Regression pipeline. Trains on startup if no saved model found, otherwise loads from disk. Designed so you can swap `predict()` for any model — transformer, ONNX, whatever — without touching the API layer.

**`backend/api/mpi_dispatcher.py`** — Partitions input across N workers and runs them in parallel using `multiprocessing.Pool`. On a real cluster, this is where you'd use `mpi4py` with `MPI.COMM_WORLD.scatter()` and `gather()`. The logic is identical — only the execution backend changes.

**`backend/cuda/cuda_inference.py`** — CUDA-accelerated softmax and batch matrix multiply using CuPy. Falls back to NumPy on Mac transparently. On an NVIDIA GPU server, the same code runs the matrix ops on device memory via cuBLAS.

**`backend/spark/spark_job.py`** — PySpark job for large batch processing. Each Spark executor runs inference on its partition independently. Run with `spark-submit` on a cluster or `local[*]` for testing.

**`frontend/src/App.jsx`** — React UI with three modes: single request, batch (MPI), and streaming (SSE). Streaming mode renders results as each worker finishes — you can watch the partitions come in.

## Running locally

**Backend:**
```bash
cd backend/api
pip install -r requirements.txt
python app.py
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

**With Docker:**
```bash
docker-compose up --build
```

Open `http://localhost:3000`. The API runs at `http://localhost:5000`.

## Testing

```bash
pip install pytest
pytest backend/tests/ -v
```

12 tests covering model loading, inference correctness, all three API endpoints, and MPI dispatch distribution.

## What I learned

The hardest part wasn't any individual component — it was getting them to compose cleanly. Flask's streaming response needs `stream_with_context` or the connection closes before the data arrives. The MPI dispatcher needs to handle uneven partitions (if you have 7 texts and 4 workers, one worker gets 3 instead of 2). The CUDA fallback needs to be truly transparent — the calling code in `app.py` should never need to check whether it's running on GPU or CPU.

The Spark integration taught me something I didn't expect: Spark's model is actually simpler than MPI for large batches because the data partitioning is automatic. MPI gives you more control but you have to manage the scatter/gather yourself. For ML inference specifically, Spark is probably the right tool for batch workloads above ~1000 records.

I built this because I wanted a project where the full stack — React, Flask, distributed compute, GPU acceleration — is all load-bearing. Every layer is doing real work.

## Tech stack

- **Frontend:** React 18, Vite, Server-Sent Events
- **API:** Flask 3, Flask-CORS
- **ML:** scikit-learn (TF-IDF + Logistic Regression)
- **Parallelism:** multiprocessing.Pool (MPI-compatible architecture, mpi4py for cluster)
- **GPU:** CuPy / CUDA (CPU fallback via NumPy)
- **Batch:** PySpark 3.5
- **Container:** Docker, docker-compose
- **CI/CD:** GitHub Actions (test → build → smoke test)
