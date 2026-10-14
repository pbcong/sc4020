import sys
import time

import faiss

from evaluate import faiss_index_bytes, load_dataset, run_benchmark

dataset = sys.argv[1] if len(sys.argv) > 1 else "coco"
data = load_dataset(dataset)

for M in [8, 16, 32, 64]:
    faiss.omp_set_num_threads(32)
    t0 = time.time()
    index = faiss.IndexPQ(data.dim, M, 8, faiss.METRIC_INNER_PRODUCT)
    index.train(data.corpus)
    index.add(data.corpus)
    build_s = time.time() - t0

    rows = run_benchmark(
        data,
        search_fn=lambda q: index.search(q, 100)[1],
        method="PQ",
        build_params=f"M={M}",
        search_params="-",
        build_s=build_s,
        index_bytes=faiss_index_bytes(index),
        csv_path=f"results/{dataset}_pq.csv",
    )
    for r in rows:
        print(f"PQ M={M:2d}  {r['query_type']:5s}  recall@100={r['recall@100']:.3f}  "
              f"qps={r['qps']:7.1f}  bytes/vec={r['bytes_per_vector']}")
