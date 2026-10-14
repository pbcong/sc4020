"""Success / failure cases on COCO: for a few captions, the exact top-10 images next to each method's top-10.

python analysis/examples.py   -> results/figures/fig12_examples.{png,pdf}
"""
import json
import os
import sys

import faiss
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "experiments"))
from ablations import add_query_edges, new_pq, train_add  # noqa: E402
from common import DATA_ROOT, SEED  # noqa: E402
from compare import COLOR, INK, INK2, save  # noqa: E402
from evaluate import load_dataset, perquery_recall  # noqa: E402

COCO_ROOT = os.path.expanduser(os.environ.get("COCO_ROOT", "~/coco2014"))
EF, PQ_M, SHOW = 100, 64, 10


def thumb(path, size=120):
    im = Image.open(os.path.join(COCO_ROOT, path)).convert("RGB")
    w, h = im.size
    s = min(w, h)
    return im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)).resize((size, size))


def main():
    data = load_dataset("coco")
    meta = json.load(open(os.path.join(DATA_ROOT, "coco", "coco_meta.json")))
    q, gt = data.queries["text"], data.gt["text"]

    faiss.omp_set_num_threads(16)
    path_idx = os.path.join(DATA_ROOT, "coco", "hnsw_M32_efc200.index")
    hnsw = faiss.read_index(path_idx)
    hnsw.hnsw.efSearch = EF
    res = {"HNSW": hnsw.search(q, 100)[1]}
    fixed = faiss.read_index(path_idx)
    add_query_edges(fixed, data.corpus, data.learn_text, 16, 16)
    fixed.hnsw.efSearch = EF
    res["HNSW + caption edges"] = fixed.search(q, 100)[1]
    pq = new_pq(data.dim, PQ_M)
    train_add(pq, data.corpus)
    res["PQ"] = pq.search(q, 100)[1]

    r10 = {m: perquery_recall(ids, gt, 10) for m, ids in res.items()}
    # captions where plain HNSW misses most of the true top-10 and the caption edges recover it,
    # plus one where PQ fails while HNSW is fine
    gain = r10["HNSW + caption edges"] - r10["HNSW"]
    picks = list(np.argsort(-gain - 0.01 * r10["HNSW + caption edges"])[:2])
    pq_fail = np.where(r10["HNSW"] >= 0.9, r10["PQ"], np.inf)
    picks.append(int(np.argmin(pq_fail)))

    rows = [("Exact", gt)] + [(m, res[m]) for m in ("HNSW", "HNSW + caption edges", "PQ")]
    colors = {"Exact": COLOR["Exact"], "HNSW": COLOR["HNSW"], "HNSW + caption edges": "#6250d6", "PQ": COLOR["PQ"]}
    fig = plt.figure(figsize=(11, 2.0 * len(picks)))
    sub = fig.subfigures(len(picks), 1)
    for sf, qi in zip(np.atleast_1d(sub), picks):
        sf.suptitle(f"“{meta['query_captions'][qi]}”", fontsize=10, color=INK, y=1.0)
        axes = sf.subplots(len(rows), SHOW + 1, gridspec_kw={"width_ratios": [1.8] + [1] * SHOW, "wspace": 0.06, "hspace": 0.08})
        true10 = set(gt[qi][:10])
        for r, (name, ids) in enumerate(rows):
            lab = axes[r, 0]
            lab.axis("off")
            txt = name if name == "Exact" else f"{name}\nrecall@10 = {r10[name][qi]:.1f}"
            lab.text(1.0, 0.5, txt, ha="right", va="center", fontsize=8, color=colors[name])
            for c in range(SHOW):
                ax = axes[r, c + 1]
                img_id = int(ids[qi][c])
                ax.imshow(thumb(meta["corpus_files"][img_id]))
                ax.set_xticks([])
                ax.set_yticks([])
                ok = img_id in true10
                for sp in ax.spines.values():
                    sp.set_visible(True)
                    sp.set_linewidth(2)
                    sp.set_color("#1baf7a" if ok else "#e34948")
    fig.text(0.5, -0.01, "Border: green = in the exact top-10, red = not in the exact top-10. "
             f"HNSW efSearch {EF}; PQ M={PQ_M}.", ha="center", fontsize=8, color=INK2)
    save(fig, "fig12_examples")


if __name__ == "__main__":
    main()
