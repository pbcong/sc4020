import argparse
import csv
import json
import os
import sys
import time

import faiss
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_ROOT  # noqa: E402
from evaluate import QUERY_TYPES, faiss_index_bytes, load_dataset, run_benchmark  # noqa: E402

# FAISS searches with max(efSearch, k), so with k = 100 any efSearch below 100 behaves like 100.
EF_SEARCH = [100, 128, 192, 256, 384, 512, 768, 1024]


def build_hnsw(corpus, M, ef_construction, threads, cache):
    meta_path = cache + ".json"
    if os.path.exists(cache) and os.path.exists(meta_path):
        print(f"loading cached index {cache}")
        with open(meta_path) as f:
            return faiss.read_index(cache), json.load(f)["build_s"]
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexHNSWFlat(corpus.shape[1], M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    t0 = time.time()
    index.add(corpus)
    build_s = time.time() - t0
    faiss.write_index(index, cache)
    with open(meta_path, "w") as f:
        json.dump({"build_s": build_s, "build_threads": threads}, f)
    return index, build_s


def dist_comps(index, queries):
    faiss.omp_set_num_threads(1)
    faiss.cvar.hnsw_stats.reset()
    index.search(queries, 100)
    return faiss.cvar.hnsw_stats.ndis / len(queries)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="coco")
    parser.add_argument("--M", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", nargs="+", type=int, default=EF_SEARCH)
    parser.add_argument("--build-threads", type=int, default=len(os.sched_getaffinity(0)))
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()

    name = args.dataset
    data = load_dataset(name)
    data.learn_text = None
    print(f"===== {name}: corpus {data.corpus.shape}, M={args.M}, efC={args.ef_construction}, "
          f"build threads {args.build_threads} =====", flush=True)
    cache = os.path.join(DATA_ROOT, name, f"hnsw_M{args.M}_efc{args.ef_construction}.index")
    index, build_s = build_hnsw(data.corpus, args.M, args.ef_construction, args.build_threads, cache)
    levels = faiss.vector_to_array(index.hnsw.levels)
    per_layer = [int((levels > l).sum()) for l in range(levels.max())]
    print(f"build {build_s:.1f}s, nodes per layer {per_layer}", flush=True)

    tag = f"M{args.M}_efc{args.ef_construction}"
    csv_path = f"results/{name}_hnsw.csv"
    extra_path = f"results/{name}_hnsw_extra.csv"
    for path in (csv_path, extra_path):
        if os.path.exists(path):
            os.remove(path)
    index_bytes = faiss_index_bytes(index)
    with open(extra_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "build_params", "efSearch", "query_type", "dist_comps_per_query", "nodes_per_layer"])
        for ef in args.ef_search:
            index.hnsw.efSearch = ef
            rows = run_benchmark(data, lambda q: index.search(q, 100)[1], method="HNSW",
                                 build_params=f"M={args.M},efC={args.ef_construction}",
                                 search_params=f"efSearch={ef}", build_s=build_s, index_bytes=index_bytes,
                                 csv_path=csv_path, runs=args.runs,
                                 perquery_path=f"results/perquery/{name}_hnsw_{tag}_ef{ef}.npz")
            ndis = {t: dist_comps(index, data.queries[t]) for t in QUERY_TYPES}
            for t in QUERY_TYPES:
                writer.writerow([name, tag, ef, t, round(ndis[t], 1), "/".join(map(str, per_layer))])
            f.flush()
            r = {row["query_type"]: row for row in rows}
            print(f"efSearch={ef:5d} R@100 img {r['image']['recall@100']:.4f} txt {r['text']['recall@100']:.4f} "
                  f"| R@10 img {r['image']['recall@10']:.4f} txt {r['text']['recall@10']:.4f} "
                  f"| qps {r['image']['qps']}/{r['text']['qps']} | ndis {ndis['image']:.0f}/{ndis['text']:.0f}",
                  flush=True)
    print(f"wrote {csv_path}, {extra_path}")


if __name__ == "__main__":
    main()
