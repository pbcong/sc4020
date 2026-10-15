"""Figures and tables for the ablations / improvements (results/ablations/*.csv).

python analysis/compare_ablations.py   -> results/figures/fig7..fig10, results/tables/ablations.md
"""
import glob
import json
import os
import re

import numpy as np
import pandas as pd

from compare import COLOR, DATASETS, INK, INK2, TITLES, md, plt, save

A = "results/ablations"
FIX = {"HNSW": COLOR["HNSW"], "HNSW+qedges": "#6250d6", "HNSW+entry": "#1baf7a", "HNSW+center": "#eda100",
       "PQ[pq]": COLOR["PQ"], "PQ[opq]": "#e87ba4", "PQ[qa-text-pq]": "#eda100", "PQ[qa-text-opq]": "#6250d6",
       "PQ[qa-mix-opq]": "#1baf7a", "PQ+rerank": "#2a78d6"}


PQ_LABEL = {"PQ[pq]": "PQ (baseline)", "PQ[opq]": "OPQ (learned rotation)",
            "PQ[qa-text-pq]": "text-weighted PQ (no rotation)", "PQ[qa-text-opq]": "text-weighted OPQ",
            "PQ[qa-mix-opq]": "image+text-weighted OPQ"}


def num(s, key):
    if not isinstance(s, str):
        return np.nan
    m = re.search(rf"{key}=([0-9.]+)", s)
    return float(m.group(1)) if m else np.nan


def load(pattern):
    files = glob.glob(f"{A}/{pattern}")
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["ef"] = df["search_params"].map(lambda s: num(s, "efSearch"))
    df["M"] = df["build_params"].map(lambda s: num(s, "M"))
    df["efC"] = df["build_params"].map(lambda s: num(s, "efC"))
    df["N"] = df["build_params"].map(lambda s: num(s, "N"))
    df["R"] = df["search_params"].map(lambda s: num(s, "R"))
    df["n_text"] = df["build_params"].map(lambda s: num(s, "n_text"))
    df["m"] = df["build_params"].map(lambda s: num(s, ",m"))
    return df


def wide(df, keys):
    w = df.pivot_table(index=keys, columns="query_type", values=["recall@100", "qps", "bytes_per_vector", "build_s"],
                       aggfunc="mean")
    w.columns = [f"{a}_{b}" for a, b in w.columns]
    w = w.reset_index()
    w["delta"] = w["recall@100_image"] - w["recall@100_text"]
    return w


def main_results(name, method):
    df = pd.read_csv(f"results/{name}_{method.lower()}.csv")
    df["ef"] = df["search_params"].map(lambda s: num(s, "efSearch"))
    df["M"] = df["build_params"].map(lambda s: num(s, "M"))
    df["method"] = method
    return df


# ------------------------------------------------------------------ Fig 7: HNSW build parameters
def fig_hnsw_params(lines):
    df = load("coco_hnsw_params_*.csv")
    base = main_results("coco", "HNSW")
    base = base[base["ef"].isin([100, 256, 512])].assign(efC=200.0)
    df = pd.concat([df, base], ignore_index=True)
    rep = df[df["build_params"].isin(["M=32,efC=200", "M=32,efC=40"])]
    w = wide(df, ["M", "efC", "ef"]).sort_values(["M", "efC", "ef"])
    spread = rep.groupby(["build_params", "search_params", "query_type"])["recall@100"].agg(["min", "max", "count"])
    lines.append("## HNSW build parameters (COCO; repeated builds averaged)\n\n" + md(w[[
        "M", "efC", "ef", "recall@100_image", "recall@100_text", "delta", "qps_text", "bytes_per_vector_image",
        "build_s_image"]], "{:.4g}") + "\n\nSpread over repeated builds (min / max / n):\n\n" + md(spread, "{:.4f}", index=True) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, (key, fixed, label) in zip(axes, [("M", ("efC", 200), "graph degree M (efConstruction = 200)"),
                                             ("efC", ("M", 32), "efConstruction (M = 32)")]):
        g = w[w[fixed[0]] == fixed[1]]
        for ef, mk in ((100, "o"), (256, "s")):
            h = g[g["ef"] == ef].sort_values(key)
            ax.plot(h[key], h["recall@100_image"], marker=mk, color=COLOR["HNSW"], label=f"image, efSearch {ef}")
            ax.plot(h[key], h["recall@100_text"], marker=mk, color=COLOR["HNSW"], ls="--", mfc="white",
                    label=f"text, efSearch {ef}")
        if key == "efC":
            r = rep[(rep["query_type"] == "text") & (rep["search_params"] == "efSearch=100")]
            for b, x in (("M=32,efC=200", 200), ("M=32,efC=40", 40)):
                v = r[r["build_params"] == b]["recall@100"]
                if len(v) > 1:
                    ax.errorbar([x], [v.mean()], yerr=[[v.mean() - v.min()], [v.max() - v.mean()]], color=INK2,
                                capsize=3, lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted(g[key].unique()), [str(int(v)) for v in sorted(g[key].unique())])
        ax.minorticks_off()
        ax.set_xlabel(label)
    axes[0].set_ylabel("recall@100")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    save(fig, "fig7_hnsw_build_params")


# ------------------------------------------------------------------ Fig 8: scaling
def fig_scaling(lines):
    df = load("laion_scaling_*.csv")
    if df.empty:
        return
    hn = main_results("laion", "HNSW")
    hn = hn[hn["ef"].isin([100, 256])].assign(N=999448.0)
    pq = main_results("laion", "PQ")
    pq = pq[pq["M"].isin([16, 64])].assign(N=999448.0)
    df = pd.concat([df, hn, pq], ignore_index=True)
    df["setting"] = np.where(df["method"] == "HNSW", "HNSW efSearch " + df["ef"].fillna(0).astype(int).astype(str),
                             "PQ M=" + df["M"].fillna(0).astype(int).astype(str))
    w = wide(df, ["setting", "N"]).sort_values(["setting", "N"])
    lines.append("## Scaling (LAION random subsets, exact top-100 recomputed per subset)\n\n" + md(w[[
        "setting", "N", "recall@100_image", "recall@100_text", "delta", "qps_image"]], "{:.4g}") + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, meth in zip(axes, ("HNSW", "PQ")):
        for setting, ls in zip(sorted(s for s in w["setting"].unique() if s.startswith(meth)), ("-", ":")):
            g = w[w["setting"] == setting]
            ax.plot(g["N"], g["recall@100_image"], marker="o", color=COLOR[meth], ls=ls, label=f"{setting}, image")
            ax.plot(g["N"], g["recall@100_text"], marker="o", color=COLOR[meth], ls=ls, mfc="white",
                    label=f"{setting}, text", alpha=0.7)
        ax.set_xscale("log")
        ax.set_xlabel("corpus size N (LAION subsets, log scale)")
        ax.set_ylabel("recall@100")
        ax.set_title(meth, color=INK)
        ax.legend(frameon=False, fontsize=7)
    save(fig, "fig8_scaling")


# ------------------------------------------------------------------ Fig 9: HNSW fixes
def fig_hnsw_fixes(lines):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9), sharey=False)
    rows = []
    for ax, name in zip(axes, DATASETS):
        base = main_results(name, "HNSW")
        parts = [base[base["ef"].isin([100, 128, 192, 256, 384, 512])]]
        q = load(f"{name}_qedges_*.csv")
        e = load(f"{name}_entry_*.csv")
        if not e.empty:
            e = e[e["m"] == 1]
        parts += [d for d in (q, e) if not d.empty]
        df = pd.concat(parts, ignore_index=True)
        n = {"coco": 122287, "laion": 999448}[name]
        df["variant"] = np.select(
            [df["method"] == "HNSW+qedges", df["method"] == "HNSW+entry"],
            ["+ caption edges (" + (100 * df["n_text"] / n).round().astype("Int64").astype(str) + "% of captions)",
             "+ caption entry points"], "HNSW (baseline)")
        for v, g in df.groupby("variant"):
            g = g.sort_values("ef")
            color = FIX.get(g["method"].iloc[0], INK2)
            alpha = 1.0 if g["method"].iloc[0] != "HNSW+qedges" or g["n_text"].iloc[0] == df["n_text"].max() else 0.45
            for qt, ls, mfc in (("image", "-", None), ("text", "--", "white")):
                h = g[g["query_type"] == qt]
                ax.plot(h["qps"], h["recall@100"], marker="o", color=color, ls=ls, mfc=mfc or color, mec=color,
                        alpha=alpha, label=v if qt == "image" else None)
        w = wide(df, ["variant", "ef"])
        w.insert(0, "dataset", name)
        rows.append(w)
        ax.set_xscale("log")
        ax.set_title(TITLES[name], color=INK)
        ax.set_xlabel("queries per second (1 thread, log scale)")
    axes[0].set_ylabel("recall@100")
    from matplotlib.lines import Line2D
    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    handles = [Line2D([], [], color=h.get_color(), marker="o", alpha=h.get_alpha(), label=l) for l, h in seen.items()]
    handles += [Line2D([], [], color=INK2, ls="-", marker="o", label="image queries"),
                Line2D([], [], color=INK2, ls="--", marker="o", mfc="white", label="text queries")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.16))
    save(fig, "fig9_hnsw_fixes")
    t = pd.concat(rows)
    lines.append("## HNSW fixes\n\n" + md(t[["dataset", "variant", "ef", "recall@100_image", "recall@100_text",
                                             "delta", "qps_image", "qps_text", "bytes_per_vector_image",
                                             "build_s_image"]], "{:.4g}") + "\n")
    for name in DATASETS:
        for f in ("textnodes", "centering"):
            d = load(f"{name}_{f}*.csv")
            if not d.empty:
                w = wide(d, ["method", "build_params", "search_params"])
                lines.append(f"## {name} {f}\n\n" + md(w, "{:.4g}") + "\n")
        p = f"{A}/{name}_centering_stats.json"
        if os.path.exists(p):
            lines.append(f"## {name} centering stats\n\n```\n{json.dumps(json.load(open(p)), indent=1)}\n```\n")


# ------------------------------------------------------------------ Fig 10: PQ fixes
def fig_pq_fixes(lines):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6), sharex=True)
    for j, name in enumerate(DATASETS):
        q = load(f"{name}_qapq_*.csv")
        r = load(f"{name}_rerank_*.csv")
        if q.empty:
            continue
        w = wide(q, ["method", "M"])
        lines.append(f"## {name} query-aware PQ\n\n" + md(w[["method", "M", "recall@100_image", "recall@100_text",
                                                             "delta", "qps_image", "bytes_per_vector_image",
                                                             "build_s_image"]], "{:.4g}") + "\n")
        errs = pd.concat([pd.read_csv(f) for f in glob.glob(f"{A}/{name}_qapq_*_score_error.csv")])
        lines.append(f"## {name} PQ score error (RMS of <q, x - x_hat> over true top-100)\n\n" +
                     md(errs.pivot_table(index=["variant", "M"], columns="query_type",
                                         values="rms_score_error_top100"), "{:.4f}", index=True) + "\n")
        for i, qt in enumerate(("image", "text")):
            ax = axes[i, j]
            for meth in ["PQ[pq]", "PQ[opq]", "PQ[qa-text-pq]", "PQ[qa-text-opq]", "PQ[qa-mix-opq]"]:
                g = w[w["method"] == meth].sort_values("M")
                if len(g):
                    ax.plot(g["M"], g[f"recall@100_{qt}"], marker="o", color=FIX[meth],
                            ls="--" if qt == "text" else "-", mfc="white" if qt == "text" else FIX[meth],
                            mec=FIX[meth], label=PQ_LABEL[meth])
            ax.set_xscale("log", base=2)
            ax.set_xticks([16, 32, 64], ["16", "32", "64"])
            ax.minorticks_off()
            ax.set_title(f"{TITLES[name]}: {qt} queries", color=INK)
            ax.set_ylabel("recall@100")
            if i == 1:
                ax.set_xlabel("M (bytes per vector)")
        if not r.empty:
            wr = wide(r, ["M", "R"])
            lines.append(f"## {name} PQ + exact re-ranking\n\n" + md(wr[["M", "R", "recall@100_image",
                                                                         "recall@100_text", "delta", "qps_image",
                                                                         "qps_text"]], "{:.4g}") + "\n")
    axes[0, 0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig10_pq_fixes")


# distinct markers, largest drawn first: median thresholds and image-mean centring coincide
LSH_VARIANTS = {"LSH": ("plain (random hyperplanes)", COLOR["LSH"], "o", 5),
                "LSH-thr": ("median thresholds", "#eda100", "s", 11),
                "LSH-center": ("image-mean centring", "#e87ba4", "^", 8.5),
                "LSH-center-QA": ("modality-aware centring (caption-aware)", "#6250d6", "o", 4)}


# ------------------------------------------------------------------ Fig 11: LSH variants
def fig_lsh_variants(lines):
    from compare import load_results
    df = load_results()
    df = df[df["method"].isin(LSH_VARIANTS)]
    if df.empty:
        return
    # text queries only: for image queries centre-QA is identical to centre by construction
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for j, name in enumerate(DATASETS):
        d = df[df["dataset"] == name]
        w = wide(d.rename(columns={"param": "nbits"}), ["method", "nbits"])
        lines.append(f"## {name} LSH variants (mean over 3 projection seeds)\n\n" + md(
            w[["method", "nbits", "recall@100_image", "recall@100_text", "delta", "qps_image",
               "bytes_per_vector_image"]], "{:.4g}") + "\n")
        ax = axes[j]
        for meth, (label, color, mk, ms) in LSH_VARIANTS.items():
            g = d[(d["method"] == meth) & (d["query_type"] == "text")].sort_values("param")
            ax.plot(g["param"], g["recall@100"], marker=mk, ms=ms, color=color, lw=1.4, mfc="white", mec=color,
                    label=label)
        ax.set_xscale("log", base=2)
        ticks = sorted(d["param"].unique())
        ax.set_xticks(ticks, [str(int(t)) for t in ticks], fontsize=7)
        ax.minorticks_off()
        ax.set_title(f"{TITLES[name]}: text queries", color=INK)
        ax.set_ylabel("recall@100")
        ax.set_xlabel("nbits (bits per vector)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    save(fig, "fig11_lsh_variants")


def main():
    lines = []
    fig_hnsw_params(lines)
    fig_scaling(lines)
    fig_hnsw_fixes(lines)
    fig_pq_fixes(lines)
    fig_lsh_variants(lines)
    os.makedirs("results/tables", exist_ok=True)
    with open("results/tables/ablations.md", "w") as f:
        f.write("\n".join(lines))
    print("wrote results/tables/ablations.md")


if __name__ == "__main__":
    main()
