"""Ablations and improvements on top of the main PQ / HNSW benchmark.

  centering  : remove each modality's mean, re-normalise, recompute ground truth  (does Δ vanish?)
  hnsw_params: HNSW build parameters M and efConstruction (COCO)
  scaling    : LAION random subsets of the corpus (10k ... 300k), exact top-100 recomputed
  rerank     : PQ + exact re-ranking of the top R = k_factor x 100 candidates
  qapq       : query-aware PQ — reweight the space by the query second moment, ± OPQ rotation
  textnodes  : HNSW + training captions inserted as extra graph nodes that are never returned (naive)
  qedges     : HNSW + query-guided edges between images that are co-neighbours of a training caption
  entry      : HNSW whose text queries start at the true nearest images of the closest training captions

Every experiment except `centering` and `scaling` keeps the original ground truth, so its rows are
directly comparable with results/<dataset>_{pq,hnsw}.csv. learn_text (training captions, never the
test queries) is used only by the query-aware methods: qapq, textnodes, entry and qedges (RULES.md rule 1).
"""
import argparse
import csv
import json
import os
import time

import faiss
import numpy as np

from common import DATA_ROOT, SEED, exact_topk, l2_normalize
from evaluate import QUERY_TYPES, Data, faiss_index_bytes, load_dataset, run_benchmark

OUT = "results/ablations"
THREADS = len(os.sched_getaffinity(0))
IP = faiss.METRIC_INNER_PRODUCT
ADD_CHUNK = 8192


def out_csv(name, tag):
    os.makedirs(OUT, exist_ok=True)
    path = f"{OUT}/{name}_{tag}.csv"
    if os.path.exists(path):
        os.remove(path)
    return path


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def build_hnsw(x, M=32, efc=200):
    faiss.omp_set_num_threads(THREADS)
    index = faiss.IndexHNSWFlat(x.shape[1], M, IP)
    index.hnsw.efConstruction = efc
    t0 = time.time()
    index.add(x)
    return index, time.time() - t0


def train_add(index, x):
    faiss.omp_set_num_threads(THREADS)
    t0 = time.time()
    index.train(x)
    for i in range(0, len(x), ADD_CHUNK):
        index.add(x[i:i + ADD_CHUNK])
    return time.time() - t0


def new_pq(d, M):
    index = faiss.IndexPQ(d, M, 8, IP)
    index.pq.cp.seed = SEED
    return index


def report(rows, extra=""):
    r = {row["query_type"]: row for row in rows}
    log(f"{r['image']['method']:16s} {r['image']['build_params']:34s} {r['image']['search_params']:14s} "
        f"R@100 img {r['image']['recall@100']:.4f} txt {r['text']['recall@100']:.4f} "
        f"Δ {r['image']['recall@100'] - r['text']['recall@100']:+.4f} | qps {r['image']['qps']}/{r['text']['qps']} "
        f"| B/vec {r['image']['bytes_per_vector']} {extra}")


def bench(data, fn, method, build, search, build_s, nbytes, path, runs):
    rows = run_benchmark(data, fn, method, build, search, build_s, nbytes, path, runs=runs)
    report(rows)
    return rows


def hnsw_sweep(data, index, build_s, method, build, path, runs, efs):
    nbytes = faiss_index_bytes(index)
    for ef in efs:
        index.hnsw.efSearch = ef
        bench(data, lambda q: index.search(q, 100)[1], method, build, f"efSearch={ef}", build_s, nbytes, path, runs)


def pq_once(data, M, method, build, path, runs):
    index = new_pq(data.dim, M)
    build_s = train_add(index, data.corpus)
    bench(data, lambda q: index.search(q, 100)[1], method, build, "-", build_s, faiss_index_bytes(index), path, runs)


def gap_stats(data):
    c = data.corpus.mean(0)
    out = {}
    for t in QUERY_TYPES:
        q = data.queries[t]
        s = np.einsum("qd,qkd->qk", q, data.corpus[data.gt[t]])
        out[t] = dict(centroid_cos=float(q.mean(0) @ c / np.linalg.norm(q.mean(0)) / np.linalg.norm(c)),
                      sim_nn1=float(s[:, 0].mean()), sim_nn100=float(s[:, 99].mean()),
                      contrast=float((s[:, 0] - s[:, 99]).mean()))
    return out


# ---------------------------------------------------------------- experiments

def centering(data, args):
    mu_i, mu_t = data.corpus.mean(0), data.learn_text.mean(0)
    corpus = l2_normalize(data.corpus - mu_i)
    queries = {"image": l2_normalize(data.queries["image"] - mu_i),
               "text": l2_normalize(data.queries["text"] - mu_t)}
    gt = {t: exact_topk(corpus, queries[t], threads=THREADS)[1] for t in QUERY_TYPES}
    overlap = {t: float(np.mean([len(set(a) & set(b)) / 100 for a, b in zip(gt[t], data.gt[t])]))
               for t in QUERY_TYPES}
    cdata = Data(name=data.name, corpus=corpus, queries=queries, gt=gt, learn_text=None)
    stats = {"before": gap_stats(data), "after": gap_stats(cdata), "gt_overlap_with_original": overlap}
    log(json.dumps(stats))
    with open(f"{OUT}/{data.name}_centering_stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    path = out_csv(data.name, "centering")
    index, build_s = build_hnsw(corpus)
    hnsw_sweep(cdata, index, build_s, "HNSW+center", "M=32,efC=200", path, args.runs, [100, 128, 256, 512])
    del index
    for M in (16, 64):
        pq_once(cdata, M, "PQ+center", f"M={M},nbits=8", path, args.runs)


def hnsw_params(data, args):
    path = out_csv(data.name, f"hnsw_params_{args.tag}")
    for spec in args.configs:
        M, efc = map(int, spec.split(","))
        index, build_s = build_hnsw(data.corpus, M, efc)
        hnsw_sweep(data, index, build_s, "HNSW", f"M={M},efC={efc}", path, args.runs, [100, 256, 512])


def scaling(data, args):
    path = out_csv(data.name, f"scaling_{args.tag}")
    for n in args.sizes:
        rows = np.sort(np.random.default_rng(SEED).choice(len(data.corpus), n, replace=False))
        corpus = np.ascontiguousarray(data.corpus[rows])
        gt = {t: exact_topk(corpus, data.queries[t], threads=THREADS)[1] for t in QUERY_TYPES}
        sub = Data(name=data.name, corpus=corpus, queries=data.queries, gt=gt, learn_text=None)
        index, build_s = build_hnsw(corpus)
        hnsw_sweep(sub, index, build_s, "HNSW", f"M=32,efC=200,N={n}", path, args.runs, [100, 256])
        del index
        for M in (16, 64):
            pq_once(sub, M, "PQ", f"M={M},nbits=8,N={n}", path, args.runs)


def rerank(data, args):
    path = out_csv(data.name, f"rerank_{args.tag}")
    for M in args.M:
        pq = new_pq(data.dim, M)
        index = faiss.IndexRefineFlat(pq)
        build_s = train_add(index, data.corpus)
        nbytes, pq_bytes = faiss_index_bytes(index), faiss_index_bytes(pq)
        for kf in args.k_factors:
            index.k_factor = kf
            bench(data, lambda q: index.search(q, 100)[1], "PQ+rerank",
                  f"M={M},pq_B/vec={pq_bytes / len(data.corpus):.1f}", f"R={kf * 100}", build_s, nbytes, path,
                  args.runs)
        del index, pq


def sqrt_inv(S, reg):
    w, V = np.linalg.eigh(S.astype(np.float64))
    w = np.maximum(w, 0) + reg * w.mean()
    return ((V * np.sqrt(w)) @ V.T).astype(np.float32), ((V / np.sqrt(w)) @ V.T).astype(np.float32)


def qapq(data, args):
    """Query-aware PQ. With A = S^(1/2), S = E[q q^T] over training queries, index x' = A x and search
    q' = A^-1 q: <q', x'> = <q, x> exactly, so the ground truth is unchanged, but PQ's squared
    reconstruction error ||x' - x'_hat||^2 = r^T S r becomes the expected squared score error for queries
    drawn from S. OPQ adds a learned rotation that balances variance across subspaces."""
    rng = np.random.default_rng(SEED)
    sample = lambda x: x[rng.choice(len(x), min(200_000, len(x)), replace=False)].astype(np.float64)
    T, I = sample(data.learn_text), sample(data.corpus)
    S = {"text": T.T @ T / len(T), "mix": (T.T @ T / len(T) + I.T @ I / len(I)) / 2}
    path = out_csv(data.name, f"qapq_{args.tag}")
    err_path = f"{OUT}/{data.name}_qapq_{args.tag}_score_error.csv"
    with open(err_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "variant", "M", "query_type", "rms_score_error_top100", "mean_contrast"])
        for variant in args.variants:
            weight, rot = {"pq": (None, False), "opq": (None, True), "qa-text-pq": ("text", False),
                           "qa-text-opq": ("text", True), "qa-mix-opq": ("mix", True)}[variant]
            if weight:
                A, Ainv = sqrt_inv(S[weight], args.reg)
            else:
                A = Ainv = np.eye(data.dim, dtype=np.float32)
            xt = np.ascontiguousarray(data.corpus @ A)
            for M in args.M:
                index = faiss.index_factory(data.dim, f"OPQ{M},PQ{M}" if rot else f"PQ{M}", IP)
                build_s = train_add(index, xt)
                fn = (lambda q: index.search(q, 100)[1]) if weight is None else \
                     (lambda q: index.search(q @ Ainv, 100)[1])
                bench(data, fn, f"PQ[{variant}]", f"M={M},nbits=8", "-", build_s, faiss_index_bytes(index),
                      path, args.runs)
                for t in QUERY_TYPES:
                    sel = rng.choice(len(data.queries[t]), min(200, len(data.queries[t])), replace=False)
                    qt = data.queries[t][sel] @ Ainv
                    xs = xt[data.gt[t][sel].ravel()]
                    xh = index.sa_decode(index.sa_encode(xs))
                    err = np.einsum("qd,qkd->qk", qt, (xs - xh).reshape(len(sel), 100, -1))
                    s = np.einsum("qd,qkd->qk", data.queries[t][sel], data.corpus[data.gt[t][sel]])
                    w.writerow([data.name, variant, M, t, float(np.sqrt((err ** 2).mean())),
                                float((s[:, 0] - s[:, 99]).mean())])
                f.flush()
                del index


def textnodes(data, args):
    """Naive routing: insert training captions as extra graph nodes, never return them."""
    N = len(data.corpus)
    n_t = int(args.text_frac * N)
    T = data.learn_text[np.sort(np.random.default_rng(SEED).choice(len(data.learn_text), n_t, replace=False))]
    X = np.vstack([data.corpus, T])
    index, build_s = build_hnsw(X)
    nbytes = faiss_index_bytes(index)
    sel = faiss.IDSelectorRange(0, N)
    path = out_csv(data.name, f"textnodes_{args.tag}")
    stats = faiss.cvar.hnsw_stats
    for ef in [100, 128, 256, 512]:
        params = faiss.SearchParametersHNSW()
        params.efSearch = ef
        params.sel = sel
        rows = run_benchmark(data, lambda q: index.search(q, 100, params=params)[1], "HNSW+textnodes",
                             f"M=32,efC=200,text_nodes={n_t}", f"efSearch={ef}", build_s, nbytes, path,
                             runs=args.runs)
        ndis = {}
        for t in QUERY_TYPES:
            faiss.omp_set_num_threads(1)
            stats.reset()
            index.search(data.queries[t], 100, params=params)
            ndis[t] = stats.ndis / len(data.queries[t])
        report(rows, f"ndis {ndis['image']:.0f}/{ndis['text']:.0f}")


def entry(data, args):
    """Anchored entry points: a text query starts its level-0 beam search at the true nearest images
    of the m training captions closest to it, instead of the result of the greedy top-down descent.
    Image queries use the normal search. Modality is detected from the vector itself."""
    N = len(data.corpus)
    rng = np.random.default_rng(SEED)
    T = np.ascontiguousarray(data.learn_text[np.sort(rng.choice(len(data.learn_text), args.n_text, replace=False))])
    t0 = time.time()
    anchors = exact_topk(data.corpus, T, k=1, threads=THREADS)[1][:, 0].copy()
    tindex, _ = build_hnsw(T, 32, 100)
    route_build_s = time.time() - t0
    mu_t, mu_i = data.learn_text.mean(0), data.corpus.mean(0)
    w, c = mu_t - mu_i, float((mu_t + mu_i) @ (mu_t - mu_i) / 2)
    detect = {t: float(np.mean((data.queries[t] @ w > c) == (t == "text"))) for t in QUERY_TYPES}
    log(f"modality detection accuracy {detect}; anchors+text graph built in {route_build_s:.1f}s")

    path_idx = os.path.join(DATA_ROOT, data.name, "hnsw_M32_efc200.index")
    index = faiss.read_index(path_idx)
    with open(path_idx + ".json") as f:
        base_build_s = json.load(f)["build_s"]
    nbytes = faiss_index_bytes(index) + faiss_index_bytes(tindex) + anchors.nbytes
    path = out_csv(data.name, f"entry_{args.tag}")
    ptr = faiss.swig_ptr

    def search(q, m):
        q = np.ascontiguousarray(q, dtype=np.float32)
        out = np.empty((len(q), 100), dtype=np.int64)
        is_text = q @ w > c
        if (~is_text).any():
            out[~is_text] = index.search(q[~is_text], 100)[1]
        if is_text.any():
            qt = np.ascontiguousarray(q[is_text])
            ep = np.ascontiguousarray(anchors[tindex.search(qt, m)[1]], dtype=np.int32)
            epd = np.ascontiguousarray(-np.einsum("qd,qmd->qm", qt, data.corpus[ep]), dtype=np.float32)
            D = np.empty((len(qt), 100), dtype=np.float32)
            I = np.empty((len(qt), 100), dtype=np.int64)
            index.search_level_0(len(qt), ptr(qt), 100, ptr(ep), ptr(epd), ptr(D), ptr(I), m, 2)
            out[is_text] = I
        return out

    tindex.hnsw.efSearch = 64
    for m in args.m:
        for ef in [100, 128, 256, 512]:
            index.hnsw.efSearch = ef
            bench(data, lambda q: search(q, m), "HNSW+entry",
                  f"M=32,efC=200,n_text={args.n_text},m={m}", f"efSearch={ef}",
                  base_build_s + route_build_s, nbytes, path, args.runs)


def add_query_edges(index, corpus, T, K, max_extra, topk=None):
    """Link the nearest image of every training caption in T to its other top-K images (both ways),
    in the free layer-0 slots of an HNSW index. Returns (edges added, mean layer-0 degree before)."""
    N = len(corpus)
    if topk is None:
        topk = exact_topk(corpus, T, k=K, threads=THREADS)[1]
    hub = np.repeat(topk[:, :1], K - 1, axis=1).ravel()
    other = topk[:, 1:].ravel()
    u, v = np.concatenate([hub, other]), np.concatenate([other, hub])
    pairs, counts = np.unique(u.astype(np.int64) * N + v, return_counts=True)
    h = index.hnsw
    nb = faiss.vector_to_array(h.neighbors)
    off = faiss.vector_to_array(h.offsets).astype(np.int64)
    width = h.cum_nb_neighbors(1)
    L0 = nb[off[:-1, None] + np.arange(width)[None, :]]
    used = (L0 >= 0).sum(1)
    existing = (np.repeat(np.arange(N, dtype=np.int64), width) * N + L0.ravel())[L0.ravel() >= 0]
    new = ~np.isin(pairs, existing)
    pairs, counts = pairs[new], counts[new]
    pu, pv = pairs // N, pairs % N
    order = np.lexsort((-counts, pu))  # by node, most-supported pair first
    pu, pv = pu[order], pv[order]
    rank = np.arange(len(pu)) - np.searchsorted(pu, pu, side="left")
    keep = rank < np.minimum(width - used[pu], max_extra)
    pu, pv, rank = pu[keep], pv[keep], rank[keep]
    nb[off[pu] + used[pu] + rank] = pv.astype(nb.dtype)
    faiss.copy_array_to_vector(nb, h.neighbors)
    return len(pu), float(used.mean())


def qedges(data, args):
    """Query-guided edges (in the spirit of RoarGraph / OOD-DiskANN): for every training caption, link its
    true nearest image to its other top-K images, both ways, using the free slots of HNSW layer 0.
    Pairs supported by more captions go first; at most max_extra new edges per node. The slots are
    already allocated, so the index size does not change. Search is the unchanged HNSW search."""
    N = len(data.corpus)
    rng = np.random.default_rng(SEED)
    T = np.ascontiguousarray(data.learn_text[np.sort(rng.choice(len(data.learn_text), args.n_text, replace=False))])
    path_idx = os.path.join(DATA_ROOT, data.name, "hnsw_M32_efc200.index")
    with open(path_idx + ".json") as f:
        base_build_s = json.load(f)["build_s"]
    index = faiss.read_index(path_idx)
    t0 = time.time()
    cache = os.path.join(DATA_ROOT, data.name, f"caption_top{args.K}_n{args.n_text}.npy")  # from gpu_topk.py
    topk = np.load(cache) if os.path.exists(cache) else None
    if topk is not None:
        log(f"using cached caption top-{args.K} from {cache}")
    n_added, deg_before = add_query_edges(index, data.corpus, T, args.K, args.max_extra, topk)
    edge_s = time.time() - t0
    log(f"added {n_added} edges ({n_added / N:.1f} per node, max {args.max_extra}) from {args.n_text} captions, "
        f"K={args.K}, in {edge_s:.1f}s; mean layer-0 degree {deg_before:.1f} -> {deg_before + n_added / N:.1f}")

    path = out_csv(data.name, f"qedges_{args.tag}")
    nbytes = faiss_index_bytes(index)
    for ef in [100, 128, 256, 512]:
        index.hnsw.efSearch = ef
        rows = run_benchmark(data, lambda q: index.search(q, 100)[1], "HNSW+qedges",
                             f"M=32,efC=200,n_text={args.n_text},K={args.K},max_extra={args.max_extra}",
                             f"efSearch={ef}", base_build_s + edge_s, nbytes, path, runs=args.runs)
        ndis = {}
        for t in QUERY_TYPES:
            faiss.omp_set_num_threads(1)
            faiss.cvar.hnsw_stats.reset()
            index.search(data.queries[t], 100)
            ndis[t] = faiss.cvar.hnsw_stats.ndis / len(data.queries[t])
        report(rows, f"ndis {ndis['image']:.0f}/{ndis['text']:.0f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("experiment", choices=["centering", "hnsw_params", "scaling", "rerank", "qapq", "textnodes", "entry", "qedges"])
    p.add_argument("--dataset", default="coco")
    p.add_argument("--tag", default="all")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--configs", nargs="+", default=["32,200"], help="hnsw_params: M,efC pairs")
    p.add_argument("--sizes", nargs="+", type=int, default=[10_000, 30_000, 100_000, 300_000])
    p.add_argument("--M", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--k-factors", nargs="+", type=int, default=[2, 5, 10])
    p.add_argument("--variants", nargs="+", default=["pq", "opq", "qa-text-pq", "qa-text-opq", "qa-mix-opq"])
    p.add_argument("--reg", type=float, default=0.01)
    p.add_argument("--text-frac", type=float, default=0.1)
    p.add_argument("--n-text", type=int, default=100_000)
    p.add_argument("--m", nargs="+", type=int, default=[1, 8])
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--max-extra", type=int, default=16)
    args = p.parse_args()
    os.makedirs(OUT, exist_ok=True)
    data = load_dataset(args.dataset)
    if args.experiment not in ("centering", "qapq", "textnodes", "entry", "qedges"):
        data.learn_text = None
    log(f"{args.experiment} on {args.dataset}: corpus {data.corpus.shape}, threads {THREADS}")
    globals()[args.experiment](data, args)
    log("done")


if __name__ == "__main__":
    main()
