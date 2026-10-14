"""
run_hnsw.py - build an index once, sweep efSearch, evaluate image + text queries.

  python run_hnsw.py --dataset coco_full --impl exact   # brute force: pipeline check + baseline
  python run_hnsw.py --dataset coco_full --impl faiss   # FAISS IndexHNSWFlat (official numbers)
  python run_hnsw.py --dataset coco_5k   --impl basic   # from-scratch hnsw_basic.py

Writes results/raw/<dataset>/<method>.csv (one row per efSearch x query_type),
<method>_perquery.csv (every query's recall and time) and <method>_meta.json.

Rules:
  * The index is built once; only efSearch changes. Image and text queries use the same index.
  * Single thread for build and search (config.THREADS).
  * recall@1/@10 and timing use k=10. FAISS searches with max(efSearch, k), so recall@100 (k=100)
    is only reported when efSearch >= 100; below that it would really run at efSearch=100.
  * Time = mean over the 1,000 queries of one query's search time, separately for image and text.
  * *_se = standard error over the 1,000 queries.
"""
import argparse
import json
import os
import time

import config as C

# Single thread everywhere, set before numpy/faiss load: otherwise OpenBLAS reserves buffers for every
# core at import (~3 GB on a 20-core laptop) and may run brute-force BLAS on several threads.
for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[var] = str(C.THREADS)

import faiss
import numpy as np
import pandas as pd

from metrics import batch_ms_per_query, index_size_mb, recall_per_query, standard_error, time_per_query

COLUMNS = ["method", "param_name", "param_value", "query_type",
           "recall@1", "recall@10", "recall@100", "qps", "index_mb", "build_s",
           "recall@1_se", "recall@10_se", "recall@100_se", "ms_per_query", "ms_per_query_se",
           "ms_per_query_batch", "dist_comps_per_query"]
METHODS = {"exact": ("Exact", "exact"), "faiss": ("HNSW", "hnsw"), "basic": ("HNSW-basic", "hnsw_basic")}


# ----------------------------------------------------------------------------- data
def load_data(name):
    f = lambda key, **kw: np.load(os.path.join(C.DATA_DIR, name, f"{key}.npy"), **kw)
    data = {
        # memory-mapped: the index keeps its own copy, so the 2 GB LAION corpus is never held twice in RAM
        "db": np.ascontiguousarray(f("db", mmap_mode="r"), dtype=np.float32),
        "image": np.ascontiguousarray(f("image_queries"), dtype=np.float32),
        "text": np.ascontiguousarray(f("text_queries"), dtype=np.float32),
        "gt_image": f("gt_image"),
        "gt_text": f("gt_text"),
        "self_ids": f("query_db_ids").astype(np.int64),
    }
    nq = len(data["image"])
    assert data["db"].shape[1] == data["image"].shape[1] == data["text"].shape[1], "dimension mismatch"
    assert len(data["text"]) == nq == len(data["gt_image"]) == len(data["gt_text"]) == len(data["self_ids"])
    print(f"{name}: db {data['db'].shape} | queries {nq} image + {nq} text | gt top-{data['gt_image'].shape[1]}")
    return data


def drop_self(I, self_ids, k):
    """Remove each image query's own corpus id from its results (by id, not by rank).
    Our queries are held out (self_ids = -1), so this removes nothing; kept for data where
    the query images are inside the corpus. Pads with -1 if fewer than k remain."""
    out = np.full((len(I), k), -1, dtype=np.int64)
    for r, (row, s) in enumerate(zip(I, self_ids)):
        row = row[(row != s) & (row >= 0)][:k]
        out[r, :len(row)] = row
    return out


def search_query_set(index, Q, qtype, k, self_ids):
    if qtype == "image":
        _, I = index.search(Q, k + 1)        # one extra slot for a possible self-match
        return drop_self(I, self_ids, k)
    _, I = index.search(Q, k)
    return I


# ----------------------------------------------------------------------------- sweep
def sweep(index, set_ef, efs, data, method, index_mb, build_s, repeats, ndis_fn=None):
    rows, perquery = [], []
    for ef in efs:
        set_ef(ef)
        times = time_per_query(index, {"image": data["image"], "text": data["text"]},
                               k=C.K, repeats=repeats, warmup=C.TIMING_WARMUP)
        for qtype in ["image", "text"]:
            Q, gt, ms = data[qtype], data[f"gt_{qtype}"], times[qtype]
            ms_batch = batch_ms_per_query(index, Q, k=C.K, repeats=repeats)
            if ndis_fn:
                ndis_fn("reset")
            I10 = search_query_set(index, Q, qtype, C.K, data["self_ids"])
            ndis = ndis_fn("read") / len(Q) if ndis_fn else np.nan
            r1, r10 = recall_per_query(I10, gt, 1), recall_per_query(I10, gt, 10)
            if ef is None or ef >= 100:
                r100 = recall_per_query(search_query_set(index, Q, qtype, 100, data["self_ids"]), gt, 100)
            else:
                r100 = np.full(len(Q), np.nan)
            rows.append({"method": method, "param_name": "efSearch" if ef else "none", "param_value": ef,
                         "query_type": qtype, "recall@1": r1.mean(), "recall@10": r10.mean(),
                         "recall@100": r100.mean(), "qps": 1e3 / ms.mean(), "index_mb": index_mb,
                         "build_s": build_s, "recall@1_se": standard_error(r1),
                         "recall@10_se": standard_error(r10), "recall@100_se": standard_error(r100),
                         "ms_per_query": ms.mean(), "ms_per_query_se": standard_error(ms),
                         "ms_per_query_batch": ms_batch, "dist_comps_per_query": ndis})
            perquery.append(pd.DataFrame({"param_value": ef, "query_type": qtype, "query": np.arange(len(Q)),
                                          "recall@1": r1, "recall@10": r10, "recall@100": r100, "ms": ms}))
            print(f"  efSearch={ef!s:>4} {qtype:5s}  R@1={r1.mean():.3f}  R@10={r10.mean():.3f}+-"
                  f"{standard_error(r10):.3f}  R@100={r100.mean():.3f}  ms/query={ms.mean():.3f}+-"
                  f"{standard_error(ms):.3f}  dist/query={ndis:6.0f}")
    return rows, pd.concat(perquery, ignore_index=True)


# ----------------------------------------------------------------------------- indexes
def build_exact(db):
    index = faiss.IndexFlatIP(db.shape[1])
    t0 = time.perf_counter()
    index.add(db)
    return index, time.perf_counter() - t0, index_size_mb(index), None


def build_faiss(db, cache=None):
    """Build IndexHNSWFlat once. If `cache` is given, the built index (+ its build time) is saved
    there and reloaded on the next run, so a sweep interrupted after a long build (LAION: ~35 min)
    does not have to rebuild. The file is the serialized index, so its size is index_mb."""
    if cache and os.path.exists(cache) and os.path.exists(cache + ".json"):
        print(f"loading cached index {cache}")
        index = faiss.read_index(cache)
        with open(cache + ".json") as fh:
            build_s = json.load(fh)["build_s"]
        mb = os.path.getsize(cache) / 1e6
    else:
        index = faiss.IndexHNSWFlat(db.shape[1], C.M_GRAPH, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = C.EF_CONSTRUCTION
        t0 = time.perf_counter()
        index.add(db)                                   # graph construction happens here
        build_s = time.perf_counter() - t0
        if cache:
            faiss.write_index(index, cache)
            with open(cache + ".json", "w") as fh:
                json.dump({"build_s": build_s, "M_graph": C.M_GRAPH, "efConstruction": C.EF_CONSTRUCTION}, fh)
            mb = os.path.getsize(cache) / 1e6
        else:
            mb = index_size_mb(index)
    levels = faiss.vector_to_array(index.hnsw.levels)
    return index, build_s, mb, [int((levels > l).sum()) for l in range(int(levels.max()))]


def build_basic(db):
    from hnsw_basic import HNSW
    # same M as FAISS: M links on layers >= 1, 2*M on layer 0, level factor 1/ln(M) - like IndexHNSWFlat
    index = HNSW(db.shape[1], M=C.M_GRAPH, ef_construction=C.EF_CONSTRUCTION)
    t0 = time.perf_counter()
    index.add(db)
    return index, time.perf_counter() - t0, index.size_mb(), index.level_histogram()


def faiss_ndis(action):
    if action == "reset":
        faiss.cvar.hnsw_stats.reset()
        return None
    return faiss.cvar.hnsw_stats.ndis


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(C.DATASETS), default="coco_full")
    ap.add_argument("--impl", choices=list(METHODS), default="faiss")
    ap.add_argument("--efs", type=int, nargs="+", default=C.EF_SWEEP, help="efSearch values to sweep")
    ap.add_argument("--cache_index", action="store_true",
                    help="faiss only: save the built index in data/<dataset>/ and reuse it on the next run")
    args = ap.parse_args()

    faiss.omp_set_num_threads(C.THREADS)
    data = load_data(args.dataset)
    method, stem = METHODS[args.impl]
    repeats = C.TIMING_REPEATS[args.impl]
    print(f"building {method} ...")
    if args.impl == "faiss":
        cache = (os.path.join(C.DATA_DIR, args.dataset, f"hnsw_M{C.M_GRAPH}_efc{C.EF_CONSTRUCTION}.faiss")
                 if args.cache_index else None)
        index, build_s, mb, layers = build_faiss(data["db"], cache)
    else:
        index, build_s, mb, layers = {"exact": build_exact, "basic": build_basic}[args.impl](data["db"])
    print(f"built in {build_s:.1f}s | index {mb:.2f} MB | nodes per layer {layers}")

    if args.impl == "exact":
        rows, perquery = sweep(index, lambda ef: None, [None], data, method, mb, build_s, repeats)
        ok = all(r["recall@1"] > 0.999 and r["recall@10"] > 0.999 and r["recall@100"] > 0.999 for r in rows)
        print("PIPELINE CHECK", "PASSED (exact search gives recall 1.0)" if ok else
              "FAILED - ground truth and evaluation do not line up")
    else:
        if args.impl == "faiss":
            set_ef, ndis_fn = (lambda ef: setattr(index.hnsw, "efSearch", ef)), faiss_ndis
        else:
            set_ef, ndis_fn = (lambda ef: setattr(index, "ef_search", ef)), None
        rows, perquery = sweep(index, set_ef, args.efs, data, method, mb, build_s, repeats, ndis_fn)

    out = os.path.join(C.RAW_DIR, args.dataset)
    os.makedirs(out, exist_ok=True)
    pd.DataFrame(rows, columns=COLUMNS).to_csv(os.path.join(out, f"{stem}.csv"), index=False)
    perquery.to_csv(os.path.join(out, f"{stem}_perquery.csv"), index=False)
    meta = dict(method=method, dataset=args.dataset, n_db=int(len(data["db"])), faiss_version=faiss.__version__,
                M_graph=None if args.impl == "exact" else C.M_GRAPH,
                efConstruction=None if args.impl == "exact" else C.EF_CONSTRUCTION,
                ef_sweep=None if args.impl == "exact" else args.efs, k=C.K, threads=C.THREADS,
                timing_repeats=repeats, nodes_per_layer=layers, build_s=build_s, index_mb=mb)
    with open(os.path.join(out, f"{stem}_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"wrote {out}{os.sep}{stem}.csv")


if __name__ == "__main__":
    main()
