import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"image": "tab:blue", "text": "tab:orange"}


def load(name):
    df = pd.read_csv(f"results/{name}_pq.csv")
    df["M"] = df["build_params"].str.extract(r"M=(\d+)").astype(int)
    return df.sort_values("M")


def plot_tradeoffs(df, name):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for row, recall in enumerate(["recall@100", "recall@10"]):
        for col, x in enumerate(["qps", "bytes_per_vector"]):
            ax = axes[row, col]
            for qtype, g in df.groupby("query_type"):
                ax.plot(g[x], g[recall], "o-", color=COLORS[qtype], label=f"{qtype} queries")
                for _, r in g.iterrows():
                    ax.annotate(f"M={r['M']}", (r[x], r[recall]), textcoords="offset points",
                                xytext=(4, 4), fontsize=8)
            ax.set_xscale("log")
            ax.set_xlabel("QPS (1 thread)" if x == "qps" else "bytes per vector")
            ax.set_ylabel(recall)
            ax.grid(True, alpha=0.3)
            ax.legend()
    fig.suptitle(f"PQ on {name}")
    fig.tight_layout()
    path = f"results/{name}_pq_tradeoffs.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def plot_delta(df, name):
    wide = df.pivot(index="M", columns="query_type", values=["recall@100", "recall@10"])
    Ms = wide.index.to_numpy()
    x = np.arange(len(Ms))
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, recall in enumerate(["recall@100", "recall@10"]):
        delta = wide[(recall, "image")] - wide[(recall, "text")]
        bars = ax.bar(x + (i - 0.5) * 0.38, delta, 0.38, label=recall)
        ax.bar_label(bars, fmt="%+.3f", fontsize=8)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x, [f"M={m}" for m in Ms])
    ax.set_ylabel("Δ recall (image − text)")
    ax.set_title(f"PQ modality gap on {name}")
    ax.legend()
    fig.tight_layout()
    path = f"results/{name}_pq_delta.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["coco", "laion"]:
        if not os.path.exists(f"results/{name}_pq.csv"):
            print(f"skip {name}: no results/{name}_pq.csv")
            continue
        df = load(name)
        plot_tradeoffs(df, name)
        plot_delta(df, name)
