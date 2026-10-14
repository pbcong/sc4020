import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_ROOT, GT_K, N_QUERIES, SEED, exact_topk, l2_normalize, write_prepared  # noqa: E402

SRC = os.path.join(DATA_ROOT, "laion", "laion-clip-512-normalized.hdf5")


def main():
    with h5py.File(SRC, "r") as f:
        print("VIBE file keys:", {k: f[k].shape for k in f.keys()}, dict(f.attrs))
        full = l2_normalize(f["train"][:])
        text = l2_normalize(f["test"][:])
        vibe_gt = f["neighbors"][:]
        learn = l2_normalize(f["learn"][:]) if "learn" in f else None

    _, our_gt = exact_topk(full, text)
    overlap = np.mean([len(set(a) & set(b)) / GT_K for a, b in zip(our_gt, vibe_gt[:, :GT_K])])
    print(f"agreement with VIBE ground truth: {overlap:.4f}")
    assert overlap > 0.999, "ground truth does not match VIBE"

    rng = np.random.default_rng(SEED)
    held_out = np.sort(rng.choice(len(full), N_QUERIES, replace=False))
    keep = np.ones(len(full), dtype=bool)
    keep[held_out] = False
    corpus = full[keep]
    image = full[held_out]

    gt_text_dist, gt_text = exact_topk(corpus, text)
    gt_image_dist, gt_image = exact_topk(corpus, image)

    arrays = {
        "corpus": corpus,
        "queries_text": text,
        "queries_image": image,
        "gt_text": gt_text,
        "gt_image": gt_image,
        "gt_text_dist": gt_text_dist,
        "gt_image_dist": gt_image_dist,
        "corpus_source_ids": np.flatnonzero(keep),
        "image_query_source_ids": held_out,
    }
    if learn is not None:
        arrays["learn_text"] = learn
    write_prepared("laion", arrays, {
        "source": "VIBE laion-clip-512-normalized (CLIP ViT-B/32, LAION-400M subset)",
        "metric": "inner_product",
        "seed": SEED,
        "vibe_gt_agreement": overlap,
    })


if __name__ == "__main__":
    main()
