"""Task 5: merge result CSVs, make the report figures and tables.

python analysis/compare.py   -> results/figures/*.png|pdf, results/tables/*.csv|md
"""
import glob
import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from evaluate import load_dataset  # noqa: E402

DATASETS = ["coco", "laion"]
TITLES = {"coco": "COCO (122k images)", "laion": "LAION (999k images)"}
METHODS = ["HNSW", "PQ", "LSH"]
COLOR = {"HNSW": "#2a78d6", "PQ": "#eb6834", "LSH": "#1baf7a", "Exact": "#52514e"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STYLE = {"image": dict(ls="-", mfc=None), "text": dict(ls="--", mfc="white")}
FIG, TAB = "results/figures", "results/tables"

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 5,
    "savefig.dpi": 200, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
})


def param_of(row):
    if row["method"] == "HNSW":
        return int(re.search(r"efSearch=(\d+)", row["search_params"]).group(1))
    if row["method"] == "PQ":
        return int(re.search(r"M=(\d+)", row["build_params"]).group(1))
    if row["method"].startswith("LSH"):
        return int(re.search(r"nbits=(\d+)", row["build_params"]).group(1))
    return 0


PARAM_NAME = {"HNSW": "efSearch", "PQ": "M (bytes/vector)", "LSH": "nbits (bits/vector)"}
PQ_FILE = {"PQ": "pq", "HNSW": "hnsw", "LSH": "lsh"}


def load_results():
    frames = []
    for name in DATASETS:
        for path in glob.glob(f"results/{name}_*.csv"):
            if path.endswith("_extra.csv"):
                continue
            frames.append(pd.read_csv(path))
    df = pd.concat(frames, ignore_index=True).drop_duplicates()
    df["param"] = df.apply(param_of, axis=1)
    # LSH is run with several projection seeds: report the mean over seeds (std kept for the tables)
    keys = ["dataset", "method", "param", "query_type"]
    num = ["recall@100", "recall@10", "qps", "bytes_per_vector", "index_mb", "build_s"]
    std = df.groupby(keys)["recall@100"].std().rename("recall@100_seed_std")
    df = df.groupby(keys, as_index=False).agg({**{c: "mean" for c in num}, "build_params": "first",
                                                "search_params": "first", "threads": "first"})
    df = df.merge(std.reset_index(), on=keys, how="left")
    return df.sort_values(keys).reset_index(drop=True)


def perquery(name, method, param):
    if method == "HNSW":
        hits = glob.glob(f"results/perquery/{name}_hnsw_*_ef{param}.npz")
    else:
        hits = glob.glob(f"results/perquery/{name}_{PQ_FILE[method]}_*{param}.npz")
        hits = [h for h in hits if re.search(rf"[A-Za-z]{param}\.npz$", h)]
    return np.load(hits[0]) if hits else None


def delta_se(pq, k, paired):
    img, txt = pq[f"image_recall@{k}"], pq[f"text_recall@{k}"]
    if paired:
        d = img - txt
        return d.std(ddof=1) / np.sqrt(len(d))
    return np.sqrt(img.var(ddof=1) / len(img) + txt.var(ddof=1) / len(txt))


def add_delta(df):
    rows = []
    for (name, method, param), g in df.groupby(["dataset", "method", "param"]):
        if method == "Exact":
            continue
        g = g.set_index("query_type")
        pq = perquery(name, method, param) if method in PQ_FILE else None
        row = dict(dataset=name, method=method, param=param,
                   qps=g.loc["image", "qps"], bytes_per_vector=g.loc["image", "bytes_per_vector"],
                   build_s=g.loc["image", "build_s"])
        for k in (10, 100):
            row[f"img@{k}"] = g.loc["image", f"recall@{k}"]
            row[f"txt@{k}"] = g.loc["text", f"recall@{k}"]
            row[f"delta@{k}"] = row[f"img@{k}"] - row[f"txt@{k}"]
            row[f"delta@{k}_se"] = delta_se(pq, k, paired=name == "coco") if pq is not None else np.nan
        rows.append(row)
    out = pd.DataFrame(rows)
    num = out.columns.difference(["dataset", "method", "param"])
    out[num] = out[num].astype(float)
    out["param"] = out["param"].astype(int)
    return out


def legend_handles(methods, with_exact=True):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COLOR[m], marker="o", label=m) for m in methods]
    if with_exact:
        h.append(Line2D([], [], color=COLOR["Exact"], marker="*", ls="none", ms=10, label="Exact (brute force)"))
    h += [Line2D([], [], color=INK2, ls="-", marker="o", label="image queries"),
          Line2D([], [], color=INK2, ls="--", marker="o", mfc="white", label="text queries")]
    return h


def save(fig, stem):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG}/{stem}.{ext}")
    plt.close(fig)
    print(f"wrote {FIG}/{stem}.png")


def fig_tradeoff(df, x, stem, xlabel, label_params=True):
    methods = [m for m in METHODS if m in set(df["method"])]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=True)
    for ax, name in zip(axes, DATASETS):
        d = df[df["dataset"] == name]
        for m in methods:
            for qt in ("image", "text"):
                g = d[(d["method"] == m) & (d["query_type"] == qt)].sort_values("param")
                ax.plot(g[x], g["recall@100"], marker="o", color=COLOR[m], ls=STYLE[qt]["ls"],
                        mfc=STYLE[qt]["mfc"] or COLOR[m], mec=COLOR[m])
                if label_params and qt == "image" and (m != "HNSW" or x == "qps"):
                    for _, r in g.iloc[[0, -1]].iterrows():
                        ax.annotate(f"{'ef' if m == 'HNSW' else 'M=' if m == 'PQ' else 'b='}{r['param']}",
                                    (r[x], r["recall@100"]), textcoords="offset points", xytext=(5, -10),
                                    fontsize=7, color=INK2)
        ex = d[(d["method"] == "Exact") & (d["query_type"] == "image")]
        if len(ex):
            ax.plot(ex[x], [1.0] * len(ex), marker="*", ms=12, color=COLOR["Exact"], ls="none")
        ax.set_xscale("log")
        ax.set_title(TITLES[name], color=INK)
        ax.set_xlabel(xlabel)
        ax.set_ylim(0, 1.03)
    axes[0].set_ylabel("recall@100")
    fig.legend(handles=legend_handles(methods), loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.13))
    save(fig, stem)


def fig_param_gap(delta):
    methods = [m for m in METHODS if m in set(delta["method"])]
    fig, axes = plt.subplots(len(DATASETS), len(methods), figsize=(4.2 * len(methods), 6.2), squeeze=False)
    for i, name in enumerate(DATASETS):
        for j, m in enumerate(methods):
            ax = axes[i, j]
            g = delta[(delta["dataset"] == name) & (delta["method"] == m)].sort_values("param")
            ax.fill_between(g["param"], g["txt@100"], g["img@100"], color=COLOR[m], alpha=0.15, lw=0)
            ax.plot(g["param"], g["img@100"], marker="o", color=COLOR[m], label="image queries")
            ax.plot(g["param"], g["txt@100"], marker="o", color=COLOR[m], ls="--", mfc="white",
                    label="text queries")
            show = g if len(g) <= 4 else g.iloc[[0, 2, 4, len(g) - 1]]
            for _, r in show.iterrows():
                ax.annotate(f"Δ {r['delta@100']:.3f}", (r["param"], (r["img@100"] + r["txt@100"]) / 2),
                            textcoords="offset points", xytext=(4, 0), fontsize=7, color=INK2, va="center")
            ax.set_xscale("log", base=2)
            ax.set_xticks(g["param"], [str(p) for p in g["param"]], fontsize=7)
            ax.minorticks_off()
            ax.set_title(f"{m} — {TITLES[name]}", color=INK)
            ax.set_xlabel(PARAM_NAME[m])
            ax.set_ylabel("recall@100")
            if i == 0 and j == 0:
                ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig4_param_sweep_gap")


def text_tax(delta):
    """HNSW: how much slower must text queries search to match the image recall at each efSearch?"""
    rows = []
    for name in DATASETS:
        g = delta[(delta["dataset"] == name) & (delta["method"] == "HNSW")].sort_values("param")
        for _, r in g.iterrows():
            target = r["img@100"]
            reach = g[g["txt@100"] >= target]
            if len(reach) == 0:
                rows.append(dict(dataset=name, efSearch_image=r["param"], image_recall=target, qps_image=r["qps"]))
                continue
            j = reach.index[0]
            k = g.index.get_loc(j)
            if k == 0:
                ef_t, qps_t = g.loc[j, "param"], g.loc[j, "qps"]
            else:  # interpolate in log(ef) / log(qps) between the two bracketing settings
                lo, hi = g.iloc[k - 1], g.loc[j]
                t = (target - lo["txt@100"]) / (hi["txt@100"] - lo["txt@100"])
                ef_t = np.exp(np.log(lo["param"]) + t * (np.log(hi["param"]) - np.log(lo["param"])))
                qps_t = np.exp(np.log(lo["qps"]) + t * (np.log(hi["qps"]) - np.log(lo["qps"])))
            rows.append(dict(dataset=name, efSearch_image=r["param"], image_recall=target, qps_image=r["qps"],
                             efSearch_text_needed=round(float(ef_t)), qps_text=float(qps_t),
                             slowdown=r["qps"] / qps_t))
    return pd.DataFrame(rows)


def neighbour_profile(data):
    out = {}
    for qt in ("image", "text"):
        q, gt = data.queries[qt], data.gt[qt]
        out[qt] = np.einsum("qd,qkd->qk", q, data.corpus[gt])
    return out


def fig_modality_gap(profiles, centre_cos):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=False)
    ranks = np.arange(1, 101)
    for ax, name in zip(axes, DATASETS):
        for qt, ls in (("image", "-"), ("text", "--")):
            s = profiles[name][qt]
            mean, lo, hi = s.mean(0), np.percentile(s, 10, axis=0), np.percentile(s, 90, axis=0)
            ax.fill_between(ranks, lo, hi, color=INK2, alpha=0.12, lw=0)
            ax.plot(ranks, mean, color=INK, ls=ls,
                    label=f"{qt} queries (cos to image centroid {centre_cos[name][qt]:.2f})")
        ax.set_xscale("log")
        ax.set_xticks([1, 10, 100], ["1", "10", "100"])
        ax.set_xlabel("rank of true neighbour")
        ax.set_title(TITLES[name], color=INK)
        ax.legend(frameon=False, loc="upper right")
    axes[0].set_ylabel("cosine similarity to query\n(mean, band = 10–90th pct)")
    save(fig, "fig1_neighbour_similarity")


def binned(x, y, edges, min_count=15):
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
    xs, ys = [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() >= min_count:
            xs.append(np.median(x[sel]))
            ys.append(y[sel].mean())
    return xs, ys


def fig_failure(delta, profiles):
    """Which queries fail? HNSW: queries far from their nearest image. PQ: queries whose neighbours are tied."""
    panels = [("HNSW", 128, "similarity of query to its nearest image (NN#1)", lambda s: s[:, 0]),
              ("PQ", 64, "neighbour contrast: sim(NN#1) − sim(NN#100)", lambda s: s[:, 0] - s[:, 99])]
    fig, axes = plt.subplots(len(DATASETS), 2, figsize=(10, 6.4), sharey="col")
    for i, name in enumerate(DATASETS):
        for j, (m, param, xlabel, feat) in enumerate(panels):
            ax = axes[i, j]
            pq = perquery(name, m, param)
            if pq is None:
                continue
            xs = {qt: feat(profiles[name][qt]) for qt in ("image", "text")}
            edges = np.quantile(np.concatenate(list(xs.values())), np.linspace(0, 1, 13))
            for qt, ls, mfc in (("image", "-", COLOR[m]), ("text", "--", "white")):
                bx, by = binned(xs[qt], pq[f"{qt}_recall@100"], edges)
                ax.plot(bx, by, marker="o", color=COLOR[m], ls=ls, mfc=mfc, mec=COLOR[m], label=f"{qt} queries")
            ax.set_title(f"{m} ({'efSearch ' if m == 'HNSW' else 'M='}{param}) — {TITLES[name]}", color=INK)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("mean recall@100 of queries in bin")
            if i == 0:
                ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    save(fig, "fig5_which_queries_fail")


def fig_hnsw_work():
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for name, mk in zip(DATASETS, ("o", "s")):
        path = f"results/{name}_hnsw_extra.csv"
        if not os.path.exists(path):
            continue
        e = pd.read_csv(path).pivot(index="efSearch", columns="query_type", values="dist_comps_per_query")
        ax.plot(e.index, e["text"] / e["image"] - 1, marker=mk, color=COLOR["HNSW"], label=TITLES[name],
                ls="-" if name == "coco" else ":")
    ax.set_xscale("log", base=2)
    ax.set_xticks(e.index, [str(v) for v in e.index], fontsize=7)
    ax.minorticks_off()
    ax.axhline(0, color=INK2, lw=0.8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("efSearch")
    ax.set_ylabel("extra distance computations\nfor text vs image queries")
    ax.legend(frameon=False)
    save(fig, "fig6_hnsw_extra_work")


def md(df, floatfmt="{:.4g}", index=False):
    if index:
        df = df.reset_index()
    fmt = lambda v: floatfmt.format(v) if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + " --- |" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def write_tables(df, delta, gap_stats):
    os.makedirs(TAB, exist_ok=True)
    df.to_csv(f"{TAB}/all_results.csv", index=False)
    delta.to_csv(f"{TAB}/delta.csv", index=False)
    pd.DataFrame(gap_stats).to_csv(f"{TAB}/modality_gap.csv", index=False)
    with open(f"{TAB}/summary.md", "w") as f:
        f.write("## Modality gap\n\n" + md(pd.DataFrame(gap_stats), "{:.3f}") + "\n\n")
        cols = ["dataset", "method", "param", "img@10", "txt@10", "img@100", "txt@100", "delta@100",
                "delta@100_se", "qps", "bytes_per_vector", "build_s"]
        f.write("## Every setting\n\n" + md(delta[cols]) + "\n\n")
        ex = df[df["method"] == "Exact"].pivot_table(index="dataset", columns="query_type", values="qps")
        f.write("## HNSW text tax\n\n" + md(text_tax(delta), "{:.3g}") + "\n\n")
        f.write("## Exact QPS\n\n" + md(ex, "{:.1f}", index=True) + "\n")
    print(f"wrote {TAB}/summary.md")


def main():
    df = load_results()
    delta = add_delta(df)
    profiles, centre_cos, gap_stats = {}, {}, []
    for name in DATASETS:
        data = load_dataset(name)
        profiles[name] = neighbour_profile(data)
        c = data.corpus.mean(0)
        centre_cos[name] = {}
        for qt in ("image", "text"):
            q = data.queries[qt]
            centre_cos[name][qt] = float(q.mean(0) @ c / np.linalg.norm(q.mean(0)) / np.linalg.norm(c))
            s = profiles[name][qt]
            gap_stats.append(dict(dataset=name, query_type=qt, cos_query_centroid_to_image_centroid=centre_cos[name][qt],
                                  mean_cos_to_corpus_centroid=float((q @ c / np.linalg.norm(c)).mean()),
                                  sim_nn1=float(s[:, 0].mean()), sim_nn10=float(s[:, 9].mean()),
                                  sim_nn100=float(s[:, 99].mean()),
                                  contrast_nn1_minus_nn100=float((s[:, 0] - s[:, 99]).mean())))
        if name == "coco":
            sims = data.queries["text"] @ data.queries["image"].T
            gap_stats.append(dict(dataset=name, query_type="matched caption-image cosine",
                                  sim_nn1=float(np.diag(sims).mean())))
        del data
    fig_modality_gap(profiles, centre_cos)
    fig_tradeoff(df, "qps", "fig2_recall_vs_qps", "queries per second (1 thread, log scale)")
    fig_tradeoff(df, "bytes_per_vector", "fig3_recall_vs_memory", "index size, bytes per vector (log scale)")
    fig_param_gap(delta)
    fig_failure(delta, profiles)
    fig_hnsw_work()
    write_tables(df, delta, gap_stats)


if __name__ == "__main__":
    main()
