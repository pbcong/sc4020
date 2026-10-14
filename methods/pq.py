import argparse
import os
import sys
import time

import faiss
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import SEED, prepared_path  # noqa: E402
from evaluate import QUERY_TYPES, Data, faiss_index_bytes, reconstruction_mse, run_benchmark  # noqa: E402

MSE_SAMPLE = 10000
ADD_CHUNK = 8192  # peak encode table: 8192 x 256 x M floats = 0.5 GB at M=64


def load_dataset(name):
    # Same as evaluate.load_dataset but skips learn_text: PQ must not train on it (RULES.md rule 1),
    # and on laion it is another 1.9 GB that does not fit alongside the corpus on an 8 GB machine.
    with h5py.File(prepared_path(name), "r") as f:
        return Data(
            name=name,
            corpus=f["corpus"][:],
            queries={t: f[f"queries_{t}"][:] for t in QUERY_TYPES},
            gt={t: f[f"gt_{t}"][:] for t in QUERY_TYPES},
            learn_text=None,
        )


def build_pq(corpus, M, nbits, threads):
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexPQ(corpus.shape[1], M, nbits, faiss.METRIC_INNER_PRODUCT)
    index.pq.cp.seed = SEED
    t0 = time.time()
    index.train(corpus)
    train_s = time.time() - t0
    # faiss encodes 256k vectors at a time with a (block x 256 x M) float table: 17 GB at M=64.
    # Adding in small chunks gives identical codes with bounded memory.
    for i in range(0, len(corpus), ADD_CHUNK):
        index.add(corpus[i:i + ADD_CHUNK])
    return index, time.time() - t0, train_s


def sweep(name, Ms, nbits, threads, runs, append, out=None):
    data = load_dataset(name)
    n, d = data.corpus.shape
    print(f"\n===== {name}: corpus {n} x {d}, k-means seed {SEED}, build threads {threads} =====")
    csv_path = out or f"results/{name}_pq.csv"
    if not append and os.path.exists(csv_path):
        os.remove(csv_path)

    sample = data.corpus[np.random.default_rng(SEED).choice(n, min(MSE_SAMPLE, n), replace=False)]
    for M in Ms:
        assert d % M == 0, f"M={M} must divide d={d}"
        index, build_s, train_s = build_pq(data.corpus, M, nbits, threads)
        index_bytes = faiss_index_bytes(index)
        rows = run_benchmark(data, lambda q: index.search(q, 100)[1], method="PQ",
                             build_params=f"M={M},nbits={nbits}", search_params="-",
                             build_s=build_s, index_bytes=index_bytes, csv_path=csv_path, runs=runs,
                             perquery_path=f"results/perquery/{name}_pq_M{M}.npz")
        mse = reconstruction_mse(sample, index.sa_decode(index.sa_encode(sample)))
        r = {row["query_type"]: row for row in rows}
        print(f"M={M:3d} ({M * nbits:4d} bits) "
              f"R@100 img {r['image']['recall@100']:.4f} txt {r['text']['recall@100']:.4f} "
              f"d={r['image']['recall@100'] - r['text']['recall@100']:+.4f} | "
              f"R@10 img {r['image']['recall@10']:.4f} txt {r['text']['recall@10']:.4f} "
              f"d={r['image']['recall@10'] - r['text']['recall@10']:+.4f} | "
              f"qps {r['image']['qps']}/{r['text']['qps']} "
              f"B/vec {r['image']['bytes_per_vector']} build {build_s:.1f}s (train {train_s:.1f}s) "
              f"mse {mse:.4f}", flush=True)
    print(f"wrote {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["coco", "laion"])
    parser.add_argument("--M", nargs="+", type=int, default=[8, 16, 32, 64])
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--build-threads", type=int, default=os.cpu_count())
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--append", action="store_true", help="append to an existing results CSV")
    parser.add_argument("--out", help="CSV path (default results/<dataset>_pq.csv); for one dataset")
    args = parser.parse_args()
    for name in args.datasets:
        sweep(name, args.M, args.nbits, args.build_threads, args.runs, args.append, args.out)


if __name__ == "__main__":
    main()
