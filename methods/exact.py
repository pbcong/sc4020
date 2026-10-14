import argparse
import os
import sys
import time

import faiss

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import faiss_index_bytes, load_dataset, run_benchmark  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="coco")
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    data = load_dataset(args.dataset)
    data.learn_text = None
    csv_path = f"results/{args.dataset}_exact.csv"
    if os.path.exists(csv_path):
        os.remove(csv_path)
    t0 = time.time()
    index = faiss.IndexFlatIP(data.dim)
    index.add(data.corpus)
    rows = run_benchmark(data, lambda q: index.search(q, 100)[1], "Exact", "-", "-",
                         time.time() - t0, faiss_index_bytes(index), csv_path, runs=args.runs)
    for r in rows:
        print(f"exact {r['query_type']:5s} recall@100={r['recall@100']:.4f} qps={r['qps']}")


if __name__ == "__main__":
    main()
