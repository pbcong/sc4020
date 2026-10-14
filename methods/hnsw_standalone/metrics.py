"""metrics.py - recall, standard error, per-query timing and index size."""
import os
import tempfile
import time

import faiss
import numpy as np


def recall_per_query(pred, gt, k=10):
    """Recall@k of every query: |returned top-k ∩ true top-k| / k. Shape (n_queries,)."""
    return np.array([len(set(p[:k]) & set(g[:k])) / k for p, g in zip(pred, gt)])


def standard_error(x):
    """Standard error of the mean over queries: sample std / sqrt(n)."""
    x = np.asarray(x, dtype=np.float64)
    return float(x.std(ddof=1) / np.sqrt(len(x)))


def delta_se(a, b, paired):
    """Standard error of mean(a) - mean(b). Paired (query i of a and b come from the same photo):
    SE of the per-query differences. Unpaired: sqrt(SE(a)^2 + SE(b)^2)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if paired:
        return standard_error(a - b)
    return float(np.hypot(standard_error(a), standard_error(b)))


def time_per_query(index, query_sets, k=10, repeats=5, warmup=200):
    """Search each query on its own and time it, for several equally sized query sets
    (e.g. {"image": Qi, "text": Qt}). The sets are interleaved query by query (image i, text i,
    image i+1, ...) so CPU clock / thermal drift affects every set equally. Each query's time
    is the median over `repeats` passes. Returns {name: per-query times in ms, shape (n,)}.
    Includes the Python->FAISS call overhead of one search() call (~10 us)."""
    names = list(query_sets)
    n = len(query_sets[names[0]])
    assert all(len(query_sets[s]) == n for s in names), "query sets must have equal size"
    for s in names:
        index.search(query_sets[s][:warmup], k)          # warm up caches
    t = {s: np.empty((repeats, n)) for s in names}
    for r in range(repeats):
        for i in range(n):
            for s in names:
                q = query_sets[s][i:i + 1]
                t0 = time.perf_counter()
                index.search(q, k)
                t[s][r, i] = time.perf_counter() - t0
    return {s: np.median(t[s], axis=0) * 1e3 for s in names}


def batch_ms_per_query(index, queries, k=10, repeats=5):
    """One search() call over all queries, total time / n, median of `repeats`. Cross-check for
    the per-call overhead in time_per_query (exact IndexFlatIP also switches to BLAS for a batch)."""
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        index.search(queries, k)
        ts.append((time.perf_counter() - t0) * 1e3 / len(queries))
    return float(np.median(ts))


def index_size_mb(index):
    """Size of the serialized FAISS index (vectors + graph links) in MB."""
    path = os.path.join(tempfile.gettempdir(), "hnsw_idx")
    faiss.write_index(index, path)
    size = os.path.getsize(path) / 1e6
    os.remove(path)
    return size
