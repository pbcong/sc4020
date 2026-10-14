"""plot_hnsw.py - HNSW figures for one dataset -> results/hnsw_plots_<dataset>.png

  python plot_hnsw.py --dataset coco_full

(a) recall@10 vs efSearch, image vs text (FAISS, plus the from-scratch implementation if it was run)
(b) recall@10 vs mean time per query, image vs text
(c) Δ recall@10 = image - text, with paired standard error
Error bars are ±1 standard error over the 1,000 queries. COCO: each image query and its caption query
come from the same photo, so SE(Δ) = std(r_img_i - r_txt_i) / sqrt(n) (paired). LAION: the text queries
are unrelated to the image queries, so SE(Δ) = sqrt(SE_img^2 + SE_txt^2)."""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import config as C
from metrics import delta_se

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=list(C.DATASETS), default="coco_full")
args = ap.parse_args()
raw = os.path.join(C.RAW_DIR, args.dataset)

df = pd.read_csv(os.path.join(raw, "hnsw.csv"))
img = df[df.query_type == "image"].set_index("param_value")
txt = df[df.query_type == "text"].set_index("param_value")
basic_path = os.path.join(raw, "hnsw_basic.csv")
basic = pd.read_csv(basic_path) if os.path.exists(basic_path) else None

pq = pd.read_csv(os.path.join(raw, "hnsw_perquery.csv"))
paired = C.PAIRED[C.source_of(args.dataset)]
w = pq.pivot_table(index=["param_value", "query"], columns="query_type", values="recall@10")
efs = w.index.get_level_values("param_value").unique()
delta = pd.Series([(w.loc[e, "image"] - w.loc[e, "text"]).mean() for e in efs], index=efs)
delta_err = pd.Series([delta_se(w.loc[e, "image"], w.loc[e, "text"], paired) for e in efs], index=efs)

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
fig.suptitle(f"HNSW (M={C.M_GRAPH}, efConstruction={C.EF_CONSTRUCTION}) on {args.dataset} "
             f"({int(pq.groupby('param_value').size().iloc[0] / 2)} queries per type, 1 thread)", fontsize=11)

# (a) recall@10 vs efSearch
ax[0].errorbar(img.index, img["recall@10"], yerr=img["recall@10_se"], fmt="o-", capsize=3, label="image queries (FAISS)")
ax[0].errorbar(txt.index, txt["recall@10"], yerr=txt["recall@10_se"], fmt="s--", capsize=3, label="text queries (FAISS)")
if basic is not None:
    for qt, mk in [("image", "^:"), ("text", "v:")]:
        b = basic[basic.query_type == qt]
        ax[0].plot(b.param_value, b["recall@10"], mk, alpha=0.6, label=f"{qt} queries (from scratch)")
ax[0].set(xscale="log", xlabel="efSearch", ylabel="recall@10 (±1 SE)", title="(a) Recall@10 vs efSearch")
ax[0].set_xticks(img.index, [str(v) for v in img.index])
ax[0].xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

# (b) recall@10 vs mean time per query
for t, fmt, name, dy in [(img, "o-", "image queries", 4), (txt, "s--", "text queries", -10)]:
    ax[1].errorbar(t["recall@10"], t["ms_per_query"], xerr=t["recall@10_se"], yerr=t["ms_per_query_se"],
                   fmt=fmt, capsize=3, label=name)
    for ef, r in t.iterrows():
        ax[1].annotate(f"ef={ef}", (r["recall@10"], r["ms_per_query"]), fontsize=7, xytext=(4, dy),
                       textcoords="offset points")
ax[1].set(yscale="log", xlabel="recall@10", ylabel="mean time per query, ms (log)", title="(b) Recall@10 vs time per query")
ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")

# (c) Δ recall@10
ax[2].bar([str(v) for v in delta.index], delta.values, yerr=delta_err.values, capsize=4)
ax[2].set(xlabel="efSearch", ylabel="Δ recall@10 (image − text, ±1 SE)", title="(c) Modality-shift penalty")
ax[2].grid(alpha=0.3, axis="y")

plt.tight_layout()
path = os.path.join(C.RESULTS_DIR, f"hnsw_plots_{args.dataset}.png")
plt.savefig(path, dpi=150)
print(f"saved {path}")
