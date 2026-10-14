import csv
import os
import time
from dataclasses import dataclass

import faiss
import h5py
import numpy as np

from common import prepared_path

QUERY_TYPES = ("image", "text")
CSV_FIELDS = [
    "dataset", "method", "build_params", "search_params", "query_type",
    "recall@100", "recall@10", "qps", "bytes_per_vector", "index_mb", "build_s", "threads",
]


@dataclass
class Data:
    name: str
    corpus: np.ndarray
    queries: dict
    gt: dict
    learn_text: np.ndarray

    @property
    def dim(self):
        return self.corpus.shape[1]


def load_dataset(name):
    with h5py.File(prepared_path(name), "r") as f:
        return Data(
            name=name,
            corpus=f["corpus"][:],
            queries={t: f[f"queries_{t}"][:] for t in QUERY_TYPES},
            gt={t: f[f"gt_{t}"][:] for t in QUERY_TYPES},
            learn_text=f["learn_text"][:] if "learn_text" in f else None,
        )


def recall_at_k(pred, gt, k):
    pred, gt = np.asarray(pred)[:, :k], np.asarray(gt)[:, :k]
    return float(np.mean([len(set(p) & set(g)) / k for p, g in zip(pred, gt)]))


def timed_search(search_fn, queries, runs=5, core=None):
    faiss.omp_set_num_threads(1)
    old_affinity = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {core if core is not None else min(old_affinity)})
    try:
        best, ids = float("inf"), None
        for _ in range(runs):
            t0 = time.perf_counter()
            ids = np.vstack([search_fn(queries[i:i + 1]) for i in range(len(queries))])
            best = min(best, time.perf_counter() - t0)
    finally:
        os.sched_setaffinity(0, old_affinity)
    return len(queries) / best, ids


def faiss_index_bytes(index):
    return int(faiss.serialize_index(index).size)


def reconstruction_mse(x, x_rec):
    return float(np.mean(np.sum((x - x_rec) ** 2, axis=1)))


def perquery_recall(pred, gt, k):
    pred, gt = np.asarray(pred)[:, :k], np.asarray(gt)[:, :k]
    return np.array([len(set(p) & set(g)) / k for p, g in zip(pred, gt)], dtype=np.float32)


def run_benchmark(data, search_fn, method, build_params, search_params,
                  build_s, index_bytes, csv_path, k=100, runs=5, perquery_path=None):
    rows, perquery = [], {}
    for qtype in QUERY_TYPES:
        qps, ids = timed_search(search_fn, data.queries[qtype], runs)
        if perquery_path:
            for kk in (10, k):
                perquery[f"{qtype}_recall@{kk}"] = perquery_recall(ids, data.gt[qtype], kk)
        rows.append({
            "dataset": data.name, "method": method,
            "build_params": build_params, "search_params": search_params,
            "query_type": qtype,
            "recall@100": round(recall_at_k(ids, data.gt[qtype], k), 5),
            "recall@10": round(recall_at_k(ids, data.gt[qtype], 10), 5),
            "qps": round(qps, 1),
            "bytes_per_vector": round(index_bytes / len(data.corpus), 2),
            "index_mb": round(index_bytes / 1e6, 3),
            "build_s": round(build_s, 3),
            "threads": 1,
        })
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new:
            writer.writeheader()
        writer.writerows(rows)
    if perquery_path:
        os.makedirs(os.path.dirname(os.path.abspath(perquery_path)), exist_ok=True)
        np.savez(perquery_path, **perquery)
    return rows


def pareto_front(df, x="qps", y="recall@100"):
    df = df.sort_values([x, y], ascending=[False, False])
    best, keep = -1.0, []
    for i, row in df.iterrows():
        if row[y] > best:
            keep.append(i)
            best = row[y]
    return df.loc[keep]
