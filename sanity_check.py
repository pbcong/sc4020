import sys
import time

import faiss
import numpy as np

from evaluate import QUERY_TYPES, faiss_index_bytes, load_dataset, recall_at_k, run_benchmark


def check(name):
    print(f"\n===== {name} =====")
    data = load_dataset(name)
    n, d = data.corpus.shape
    print(f"corpus {n} x {d}; queries " + ", ".join(f"{t}={len(q)}" for t, q in data.queries.items()))

    norms = np.linalg.norm(data.corpus[:10000], axis=1)
    print(f"corpus norms: min {norms.min():.4f} max {norms.max():.4f}")

    t0 = time.time()
    index = faiss.IndexFlatIP(d)
    index.add(data.corpus)
    rows = run_benchmark(data, lambda q: index.search(q, 100)[1], "Exact", "-", "-",
                         time.time() - t0, faiss_index_bytes(index), f"results/{name}_exact.csv", runs=1)
    for r in rows:
        print(f"exact  {r['query_type']:5s} recall@100={r['recall@100']:.4f} recall@10={r['recall@10']:.4f} qps={r['qps']}")
        assert r["recall@100"] > 0.999, "exact search must reach recall 1.0"

    rng = np.random.default_rng(0)
    for t in QUERY_TYPES:
        rand = rng.integers(0, n, size=(len(data.queries[t]), 100))
        print(f"random {t:5s} recall@100={recall_at_k(rand, data.gt[t], 100):.5f} (expected ~{100 / n:.5f})")

    img_c = data.corpus.mean(0)
    for t in QUERY_TYPES:
        q = data.queries[t]
        top1 = np.einsum("ij,ij->i", q, data.corpus[data.gt[t][:, 0]])
        top100 = np.einsum("ij,ij->i", q, data.corpus[data.gt[t][:, 99]])
        centre_cos = q.mean(0) @ img_c / np.linalg.norm(q.mean(0)) / np.linalg.norm(img_c)
        print(f"{t:5s} queries: cos to corpus centre {centre_cos:.3f}, "
              f"sim to NN#1 {top1.mean():.3f}, to NN#100 {top100.mean():.3f}")

    if name == "coco":
        sims = data.queries["text"] @ data.queries["image"].T
        ranks = (sims > np.diag(sims)[:, None]).sum(1)
        r = {k: np.mean(ranks < k) for k in (1, 5, 10)}
        print(f"CLIP text->image on 1000 held-out pairs: R@1 {r[1]:.3f} R@5 {r[5]:.3f} R@10 {r[10]:.3f}")
        print(f"matched caption-image cosine: {np.diag(sims).mean():.3f}")


if __name__ == "__main__":
    faiss.omp_set_num_threads(1)
    for name in sys.argv[1:] or ["coco", "laion"]:
        check(name)
    print("\nALL CHECKS PASSED")
