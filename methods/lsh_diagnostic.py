"""Why does plain LSH lose so much more recall on text queries than on image queries? A mechanism check.

Part 1, collision: for (query, corpus vector) pairs at exact ranks 1, 10, 100, 1000, 10000 plus random corpus
vectors, compare the per-bit collision rate of the seeded sign projections with Charikar's
P[bit equal] = 1 - theta/pi. If both modalities sit on the same curve, the hashing treats a text pair and an
image pair with the same cosine identically, so the recall gap must come from where the neighbours sit.

Part 2, separation: per random hyperplane, D = [bit(q) != bit(n)] - [bit(q) != bit(f)] for the 100th true
neighbour n and the vector f at exact rank `far` has E[D] = (theta_qf - theta_qn) / pi and, since a hyperplane
gives D != 0 exactly when it separates n from f, Var[D] = theta_nf / pi - E[D]^2. Over nbits independent
hyperplanes z = E[D] sqrt(nbits / Var[D]) says how many noise std n is ahead of f in Hamming distance, and
Phi(-z) is the chance the hashes put f ahead of n. The empirical rate from the seed-0 projections of every
nbits (ties count half) is reported next to it; it can sit below the model because the rows of a random
rotation are orthogonal rather than independent.

Projections are the same as methods/lsh.py's plain variant (RandomRotationMatrix seeded with common.SEED).
Test queries are only measured here, nothing is fitted on them. Outputs go to results/tables/ so a glob-merge
of the benchmark CSVs does not pick them up.
"""
import argparse
import math
import os
import sys

import faiss
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import SEED  # noqa: E402
from evaluate import QUERY_TYPES, load_dataset  # noqa: E402
from lsh import NBITS  # noqa: E402
from lsh_tables import load  # noqa: E402

RANKS = [1, 10, 100, 1000, 10000]  # exact ranks probed per query (1-based)
NEAR_RANK = 100                     # the last true neighbour counted by recall@100
FAR_RANKS = [1000, 10000]
PROBE_NBITS = 4096                  # projections used for the collision curve (finer per-pair estimate)
N_RANDOM = 20                       # random corpus vectors per query, to cover low cosines
BIN_WIDTH, MIN_BIN = 0.05, 30
BATCH = 100

COLORS = {"image": "#2a78d6", "text": "#eb6834"}  # reference palette slots 1-2, light surface
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def sign_bits(x, nbits, seed=SEED):
    """One bit per projection, set where the projection is positive (as IndexLSH without thresholds)."""
    rot = faiss.RandomRotationMatrix(x.shape[1], nbits)
    rot.init(seed)
    return rot.apply(np.ascontiguousarray(x, dtype=np.float32)) > 0


def exact_ranks(corpus, queries):
    """Corpus ids and cosines at the exact RANKS, per query."""
    kth = [r - 1 for r in RANKS]
    ids = np.empty((len(queries), len(RANKS)), np.int64)
    for i in range(0, len(queries), BATCH):
        sims = queries[i:i + BATCH] @ corpus.T
        ids[i:i + BATCH] = np.argpartition(-sims, kth, axis=1)[:, kth]
    return ids, np.einsum("ij,ikj->ik", queries, corpus[ids])


def collision_pairs(corpus, queries, rank_ids, rng):
    rand = rng.integers(0, len(corpus), size=(len(queries), N_RANDOM))
    ids = np.concatenate([rank_ids, rand], axis=1)
    cos = np.einsum("ij,ikj->ik", queries, corpus[ids])
    uniq, inv = np.unique(ids.ravel(), return_inverse=True)
    bits_c = sign_bits(corpus[uniq], PROBE_NBITS)[inv.reshape(ids.shape)]
    collision = (sign_bits(queries, PROBE_NBITS)[:, None, :] == bits_c).mean(axis=2)
    labels = np.array([str(r) for r in RANKS] + ["random"] * N_RANDOM)
    return pd.DataFrame({"rank": np.broadcast_to(labels, ids.shape).ravel(),
                         "cos": cos.ravel(), "collision": collision.ravel()})


def bin_collisions(pairs):
    pairs = pairs.assign(theory=1 - np.arccos(np.clip(pairs["cos"], -1, 1)) / np.pi,
                         bin=np.floor(pairs["cos"] / BIN_WIDTH) * BIN_WIDTH)
    out = pairs.groupby(["query_type", "bin"]).agg(n=("cos", "size"), cos=("cos", "mean"),
                                                   collision=("collision", "mean"), theory=("theory", "mean"))
    return out[out["n"] >= MIN_BIN].reset_index()


def separation(corpus, queries, rank_ids, rank_sims, nbits_list):
    p = np.arccos(np.clip(rank_sims, -1, 1)) / np.pi  # P[bit differs] per probed rank
    near = RANKS.index(NEAR_RANK)
    rows = []
    for nbits in nbits_list:
        bq = sign_bits(queries, nbits)
        hamming = {r: (bq != sign_bits(corpus[rank_ids[:, RANKS.index(r)]], nbits)).sum(1)
                   for r in [NEAR_RANK] + FAR_RANKS}
        for far_rank in FAR_RANKS:
            far = RANKS.index(far_rank)
            pn, pf = p[:, near], p[:, far]
            cos_nf = np.einsum("ij,ij->i", corpus[rank_ids[:, near]], corpus[rank_ids[:, far]])
            p_nf = np.arccos(np.clip(cos_nf, -1, 1)) / np.pi
            z = (pf - pn) * np.sqrt(nbits / (p_nf - (pf - pn) ** 2))
            hn, hf = hamming[NEAR_RANK], hamming[far_rank]
            rows.append({
                "nbits": nbits, "far_rank": far_rank,
                f"cos_{NEAR_RANK}": np.median(rank_sims[:, near]), "cos_far": np.median(rank_sims[:, far]),
                "gap_deg": np.median((pf - pn) * 180),
                "z": np.median(z),
                "p_swap_model": np.mean([0.5 * math.erfc(v / math.sqrt(2)) for v in z]),
                "p_swap_seed0": np.mean((hf < hn) + 0.5 * (hf == hn)),
            })
    return pd.DataFrame(rows)


def plot(name, binned, sep, cos_by_rank, path):
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK_2, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK})
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.3), facecolor=SURFACE)
    for ax in (a, b):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    c = np.linspace(-0.2, 1, 200)
    a.plot(c, 1 - np.arccos(c) / np.pi, color=INK_2, linewidth=1.5, linestyle="--", label="theory 1 - θ/π")
    for qtype in QUERY_TYPES:
        lo, hi = cos_by_rank.loc[qtype, str(NEAR_RANK)], cos_by_rank.loc[qtype, "1"]
        a.axvspan(lo, hi, color=COLORS[qtype], alpha=0.12, linewidth=0)
        a.text((lo + hi) / 2, 0.42, f"{qtype}\ntop-100", ha="center", va="bottom", fontsize=8, color=INK_2)
        d = binned[binned["query_type"] == qtype]
        a.plot(d["cos"], d["collision"], "o", markersize=5, color=COLORS[qtype],
               markeredgecolor=SURFACE, markeredgewidth=1, label=f"{qtype} queries")
    a.set_xlabel("cosine(query, corpus vector)")
    a.set_ylabel(f"share of equal bits ({PROBE_NBITS} seed-{SEED} projections)")
    a.set_title("(a) Per-bit collision vs cosine; shading = median top-100 range", loc="left", fontsize=9.5)
    a.set_ylim(0.4, 1.0)
    a.legend(frameon=False, loc="upper left")

    far = FAR_RANKS[0]
    for qtype in QUERY_TYPES:
        d = sep[(sep["query_type"] == qtype) & (sep["far_rank"] == far)]
        b.plot(d["nbits"], d["p_swap_model"], color=COLORS[qtype], linewidth=2, label=f"{qtype} queries (model)")
        b.plot(d["nbits"], d["p_swap_seed0"], "o", markersize=5, color=COLORS[qtype],
               markeredgecolor=SURFACE, markeredgewidth=1)
        last = d.iloc[-1]
        b.annotate(qtype, (last["nbits"], last["p_swap_model"]), xytext=(6, 0), textcoords="offset points",
                   va="center", fontsize=8.5, color=INK_2)
    b.set_xscale("log", base=2)
    b.set_xticks(sorted(sep["nbits"].unique()), [str(n) for n in sorted(sep["nbits"].unique())])
    b.set_ylim(0, 0.5)
    b.set_xlabel("nbits")
    b.set_ylabel(f"P(rank-{far} vector hashed closer than rank-{NEAR_RANK})")
    b.set_title(f"(b) Swap probability; line = random-hyperplane model, dots = seed {SEED}",
                loc="left", fontsize=9.5)
    b.legend(frameon=False, loc="upper right")

    fig.suptitle(f"{name}: plain LSH, image vs text queries", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", nargs="+", default=["coco"])
    parser.add_argument("--nbits", nargs="+", type=int, default=NBITS)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    for name in args.dataset:
        data = load_dataset(name)
        rng = np.random.default_rng(SEED)
        pairs, seps, cos_by_rank = [], [], {}
        for qtype in QUERY_TYPES:
            q = data.queries[qtype]
            rank_ids, rank_sims = exact_ranks(data.corpus, q)
            gt_sims = np.einsum("ij,ij->i", q, data.corpus[data.gt[qtype][:, NEAR_RANK - 1]])
            assert np.allclose(rank_sims[:, RANKS.index(NEAR_RANK)], gt_sims, atol=1e-4), "exact ranks disagree with gt"
            pairs.append(collision_pairs(data.corpus, q, rank_ids, rng).assign(query_type=qtype))
            seps.append(separation(data.corpus, q, rank_ids, rank_sims, args.nbits).assign(query_type=qtype))
            cos_by_rank[qtype] = dict(zip(map(str, RANKS), np.median(rank_sims, axis=0)))
        pairs, sep = pd.concat(pairs), pd.concat(seps, ignore_index=True)
        cos_by_rank = pd.DataFrame(cos_by_rank).T
        cos_by_rank["random"] = pairs[pairs["rank"] == "random"].groupby("query_type")["cos"].median()

        df = load(name, args.results_dir, suffixes=["lsh"])
        if df is not None:
            measured = df.groupby(["query_type", "nbits"])["recall@100"].mean()
            sep = sep.join(measured.rename("recall@100_measured"), on=["query_type", "nbits"])
        sep.insert(0, "dataset", name)
        sep = sep[["dataset", "query_type"] + [c for c in sep.columns if c not in ("dataset", "query_type")]]

        print(f"\n===== {name}: median cosine of the query to its exact rank-k corpus vector =====")
        print(cos_by_rank.round(4).to_string())

        binned = bin_collisions(pairs)
        dev = (binned["collision"] - binned["theory"]).abs().groupby(binned["query_type"]).max()
        print(f"\n===== {name}: collision vs 1 - theta/pi ({PROBE_NBITS} projections, bins of {BIN_WIDTH}) =====")
        print("max |empirical - theory| per modality: " + ", ".join(f"{k} {v:.4f}" for k, v in dev.items()))

        print(f"\n===== {name}: separation of the {NEAR_RANK}th true neighbour from farther vectors =====")
        print(sep.round(4).to_string(index=False))

        out_dir = os.path.join(args.results_dir, "tables")
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.join(out_dir, f"{name}_lsh_diagnostic")
        sep.round(5).to_csv(f"{stem}.csv", index=False)
        binned.round(5).to_csv(f"{stem}_collision.csv", index=False)
        plot(name, binned, sep, cos_by_rank, f"{stem}.png")
        print(f"wrote {stem}.csv, {stem}_collision.csv, {stem}.png")


if __name__ == "__main__":
    main()
