"""
hnsw_basic.py - a basic, from-scratch HNSW (Hierarchical Navigable Small World) index.

Written directly from the pseudocode in:
  Y. A. Malkov & D. A. Yashunin, "Efficient and robust approximate nearest neighbor
  search using Hierarchical Navigable Small World graphs", IEEE TPAMI 42(4), 2020.
  (arXiv:1603.09320)

  Algorithm 1  INSERT                       -> HNSW._insert
  Algorithm 2  SEARCH-LAYER                 -> HNSW._search_layer
  Algorithm 3  SELECT-NEIGHBORS-SIMPLE      -> HNSW._select_simple
  Algorithm 4  SELECT-NEIGHBORS-HEURISTIC   -> HNSW._select_heuristic
  Algorithm 5  K-NN-SEARCH                  -> HNSW.search

Purpose in this project: a readable reference to explain HNSW in the report and to
sanity-check FAISS IndexHNSWFlat. The official numbers come from FAISS (run_hnsw.py).

Similarity is inner product (vectors are L2-normalised, so IP = cosine). Internally we
use distance = -IP so that "smaller = closer", exactly as in the paper.

The API mimics FAISS: index.add(X), index.search(Q, k) -> (similarities, ids).
"""
import heapq
import math
import random

import numpy as np


class HNSW:
    def __init__(self, dim, M=16, ef_construction=100, use_heuristic=True, seed=42):
        self.dim = dim
        self.M = M                      # links per new node, and max links on layers >= 1
        self.M0 = 2 * M                 # max links on layer 0 (paper recommends 2*M)
        self.ef_construction = ef_construction
        self.ef_search = 50             # set this before calling search (like FAISS efSearch)
        self.mL = 1.0 / math.log(M)     # level-generation factor, paper's optimum 1/ln(M)
        self.use_heuristic = use_heuristic
        self.rng = random.Random(seed)

        self.data = np.empty((0, dim), dtype=np.float32)
        self.ntotal = 0
        self.graph = []                 # graph[layer] = {node_id: [neighbour ids]}
        self.entry_point = None
        self.max_level = -1

    # ------------------------------------------------------------------ helpers
    def _dist(self, q, ids):
        """Distances from q to many nodes at once (one matrix-vector product)."""
        return -(self.data[ids] @ q)

    def _random_level(self):
        # l = floor(-ln(U) * mL), U ~ Uniform(0, 1]: exponentially fewer nodes per layer
        return int(-math.log(1.0 - self.rng.random()) * self.mL)

    # ------------------------------------------------------- Algorithm 2
    def _search_layer(self, q, entry_points, ef, layer):
        """Greedy best-first search on one layer.
        Returns up to `ef` nearest found, as a list of (dist, id) sorted ascending."""
        visited = set(entry_points)
        d0 = self._dist(q, entry_points).tolist()
        candidates = list(zip(d0, entry_points))            # min-heap: closest first
        heapq.heapify(candidates)
        W = [(-d, e) for d, e in sorted(zip(d0, entry_points))[:ef]]  # max-heap: furthest first
        heapq.heapify(W)

        links = self.graph[layer]
        while candidates:
            dc, c = heapq.heappop(candidates)
            if dc > -W[0][0]:          # closest candidate is worse than furthest result
                break                  # -> nothing left can improve W
            nbrs = [n for n in links[c] if n not in visited]
            if not nbrs:
                continue
            visited.update(nbrs)
            for d, n in zip(self._dist(q, nbrs).tolist(), nbrs):
                if len(W) < ef or d < -W[0][0]:
                    heapq.heappush(candidates, (d, n))
                    heapq.heappush(W, (-d, n))
                    if len(W) > ef:
                        heapq.heappop(W)       # drop the furthest
        return sorted((-nd, n) for nd, n in W)

    # ------------------------------------------------------- Algorithms 3 & 4
    @staticmethod
    def _select_simple(cands, M):
        """cands: list of (dist, id) sorted ascending. Keep the M closest."""
        return [e for _, e in cands[:M]]

    def _select_heuristic(self, cands, M):
        """Keep e only if it is closer to the base point than to every neighbour already
        kept. Spreads links in different directions, which keeps the graph navigable
        on clustered data (like CLIP embeddings)."""
        selected = []
        for d_q, e in cands:
            if len(selected) >= M:
                break
            if not selected or np.all(d_q < -(self.data[selected] @ self.data[e])):
                selected.append(e)
        return selected

    def _select(self, cands, M):
        return self._select_heuristic(cands, M) if self.use_heuristic else self._select_simple(cands, M)

    # ------------------------------------------------------- Algorithm 1
    def _insert(self, idx):
        q = self.data[idx]
        level = self._random_level()
        while len(self.graph) <= level:
            self.graph.append({})
        for l in range(level + 1):
            self.graph[l][idx] = []

        if self.entry_point is None:                       # first node
            self.entry_point, self.max_level = idx, level
            return

        # Phase 1: greedy descent (ef = 1) through layers above the new node's level
        ep = [self.entry_point]
        for l in range(self.max_level, level, -1):
            ep = [self._search_layer(q, ep, 1, l)[0][1]]

        # Phase 2: on each layer the node lives in, find ef_construction candidates,
        # link to M of them, and link back (shrinking neighbours that overflow)
        for l in range(min(level, self.max_level), -1, -1):
            W = self._search_layer(q, ep, self.ef_construction, l)
            neighbours = self._select(W, self.M)
            self.graph[l][idx] = list(neighbours)
            M_max = self.M0 if l == 0 else self.M
            for n in neighbours:
                n_links = self.graph[l][n]
                n_links.append(idx)
                if len(n_links) > M_max:
                    dn = self._dist(self.data[n], n_links)
                    order = np.argsort(dn)
                    cand = [(float(dn[i]), n_links[i]) for i in order]
                    self.graph[l][n] = self._select(cand, M_max)
            ep = [e for _, e in W]

        if level > self.max_level:
            self.entry_point, self.max_level = idx, level

    def add(self, X):
        X = np.ascontiguousarray(X, dtype=np.float32)
        start = self.ntotal
        self.data = np.vstack([self.data, X])
        self.ntotal = len(self.data)
        for idx in range(start, self.ntotal):
            self._insert(idx)

    # ------------------------------------------------------- Algorithm 5
    def search(self, Q, k):
        Q = np.atleast_2d(np.ascontiguousarray(Q, dtype=np.float32))
        ef = max(self.ef_search, k)                        # same rule as FAISS
        D = np.full((len(Q), k), -np.inf, dtype=np.float32)
        I = np.full((len(Q), k), -1, dtype=np.int64)
        for qi, q in enumerate(Q):
            ep = [self.entry_point]
            for l in range(self.max_level, 0, -1):
                ep = [self._search_layer(q, ep, 1, l)[0][1]]
            for j, (d, n) in enumerate(self._search_layer(q, ep, ef, 0)[:k]):
                D[qi, j], I[qi, j] = -d, n
        return D, I

    # ------------------------------------------------------- stats
    def size_mb(self):
        """Vectors (float32) + links (int32), the same things FAISS stores."""
        n_links = sum(len(v) for layer in self.graph for v in layer.values())
        return (self.data.nbytes + 4 * n_links) / 1e6

    def level_histogram(self):
        return [len(layer) for layer in self.graph]


# ---------------------------------------------------------------- self-test
if __name__ == "__main__":
    import time

    rng = np.random.default_rng(0)
    n, nq, dim, k = 3000, 200, 64, 10
    X = rng.standard_normal((n, dim)).astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Q = rng.standard_normal((nq, dim)).astype(np.float32)
    Q /= np.linalg.norm(Q, axis=1, keepdims=True)
    gt = np.argsort(-(Q @ X.T), axis=1)[:, :k]             # brute force

    t0 = time.perf_counter()
    index = HNSW(dim, M=16, ef_construction=100)
    index.add(X)
    print(f"built {n} vectors in {time.perf_counter() - t0:.1f}s, "
          f"nodes per layer = {index.level_histogram()}")
    for ef in [10, 20, 50, 100, 200]:
        index.ef_search = ef
        _, I = index.search(Q, k)
        rec = np.mean([len(set(a) & set(b)) / k for a, b in zip(I, gt)])
        print(f"efSearch={ef:4d}  recall@{k}={rec:.3f}")
