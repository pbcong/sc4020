"""Exact top-K images for a sample of training captions on one GPU (CuPy, fp32), cached for ablations.py qedges.

CUDA_VISIBLE_DEVICES=6 python data_prep/gpu_topk.py --dataset laion --n-text 999448 --K 16
"""
import argparse
import os
import sys
import time

import cupy as cp
import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import DATA_ROOT, SEED, prepared_path  # noqa: E402


def cache_path(name, n_text, K):
    return os.path.join(DATA_ROOT, name, f"caption_top{K}_n{n_text}.npy")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="laion")
    p.add_argument("--n-text", type=int, required=True)
    p.add_argument("--K", type=int, default=16)
    p.add_argument("--block", type=int, default=2048)
    p.add_argument("--limit", type=int, default=None, help="only the first N selected captions (for checking)")
    args = p.parse_args()

    with h5py.File(prepared_path(args.dataset), "r") as f:
        corpus = f["corpus"][:]
        learn = f["learn_text"][:]
    rows = np.sort(np.random.default_rng(SEED).choice(len(learn), args.n_text, replace=False))  # as ablations.qedges
    T = learn[rows[:args.limit]] if args.limit else learn[rows]
    del learn
    t0 = time.time()
    db = cp.asarray(corpus)
    out = np.empty((len(T), args.K), dtype=np.int64)
    for start in range(0, len(T), args.block):
        s = cp.asarray(T[start:start + args.block]) @ db.T
        part = cp.argpartition(-s, args.K, axis=1)[:, :args.K]
        order = cp.argsort(-cp.take_along_axis(s, part, axis=1), axis=1)
        out[start:start + len(part)] = cp.take_along_axis(part, order, axis=1).get()
        if start // args.block % 100 == 0:
            print(f"  {start + len(part)}/{len(T)} in {time.time() - t0:.0f}s", flush=True)
    path = cache_path(args.dataset, args.n_text, args.K) if not args.limit else path_check(args)
    np.save(path, out)
    print(f"top-{args.K} for {len(T)} captions vs {len(corpus)} images in {time.time() - t0:.1f}s -> {path}")


def path_check(args):
    return os.path.join(DATA_ROOT, args.dataset, f"caption_top{args.K}_n{args.n_text}_first{args.limit}.npy")


if __name__ == "__main__":
    main()
