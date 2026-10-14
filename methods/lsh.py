"""LSH (random-projection sign hashing) sweep over nbits, on one index per setting.

Variants (each writes results/<dataset>_<suffix>.csv):
  plain      LSH            random hyperplanes through the origin, nothing learned
  thr        LSH-thr        per-bit thresholds = median of the projected corpus (data-dependent;
                            fitted on a seeded sample of THR_TRAIN corpus vectors)
  center     LSH-center     corpus image mean subtracted from corpus and all queries
  center_qa  LSH-center-QA  QUERY-AWARE: as `center`, but text queries are centred on the text mean
                            estimated from `learn_text` (never the test queries)

Projections come from faiss.RandomRotationMatrix seeded with common.SEED + i (i < --seeds);
faiss.IndexLSH's built-in rotation uses a hardcoded seed, so we build it ourselves.
Recall is always scored against the original ground truth.
"""
import argparse
import dataclasses
import os
import sys
import time

import faiss
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import SEED  # noqa: E402
from evaluate import faiss_index_bytes, load_dataset, run_benchmark  # noqa: E402

NBITS = [64, 128, 256, 512, 1024, 2048, 4096]
THR_TRAIN = 100_000  # corpus vectors used to fit the per-bit median thresholds (thr variant)
ADD_CHUNK = 20_000
VARIANTS = {
    "plain": ("LSH", "lsh"),
    "thr": ("LSH-thr", "lsh_thr"),
    "center": ("LSH-center", "lsh_center"),
    "center_qa": ("LSH-center-QA", "lsh_center_qa"),
}


def build_index(corpus, nbits, seed, variant):
    d = corpus.shape[1]
    rot = faiss.RandomRotationMatrix(d, nbits)
    rot.init(seed)
    index = faiss.IndexPreTransform(rot, faiss.IndexLSH(nbits, nbits, False, variant == "thr"))
    if variant in ("center", "center_qa"):
        index.prepend_transform(faiss.CenteringTransform(d))
        index.train(corpus)  # corpus mean, in the input space
    if variant == "thr":
        # thresholds are fitted in the projected space (n x nbits floats), so use a corpus sample
        rows = np.sort(np.random.default_rng(seed).choice(len(corpus), min(len(corpus), THR_TRAIN), replace=False))
        index.train(corpus[rows])
    # IndexPreTransform projects everything it is given at once; chunk to bound memory at large nbits
    for i in range(0, len(corpus), ADD_CHUNK):
        index.add(corpus[i:i + ADD_CHUNK])
    return index


def query_aware_data(data):
    """Shift text queries so that, after the index subtracts the image mean, they are text-mean centred."""
    shift = data.corpus.mean(0) - data.learn_text.mean(0)
    queries = dict(data.queries, text=np.ascontiguousarray(data.queries["text"] + shift, dtype=np.float32))
    return dataclasses.replace(data, queries=queries)


def sweep(data, variant, nbits_list, seeds, runs, out_dir, append):
    method, suffix = VARIANTS[variant]
    csv_path = os.path.join(out_dir, f"{data.name}_{suffix}.csv")
    if not append and os.path.exists(csv_path):
        os.remove(csv_path)
    for nbits in nbits_list:
        for seed in seeds:
            faiss.omp_set_num_threads(1)
            t0 = time.perf_counter()
            index = build_index(data.corpus, nbits, seed, variant)
            eval_data = query_aware_data(data) if variant == "center_qa" else data
            build_s = time.perf_counter() - t0
            rows = run_benchmark(eval_data, lambda q: index.search(q, 100)[1],
                                 method=method, build_params=f"nbits={nbits};seed={seed}", search_params="-",
                                 build_s=build_s, index_bytes=faiss_index_bytes(index),
                                 csv_path=csv_path, runs=runs)
            img, txt = rows
            print(f"{data.name} {method:14s} nbits={nbits:5d} seed={seed} "
                  f"R@100 img {img['recall@100']:.4f} txt {txt['recall@100']:.4f} "
                  f"(delta {img['recall@100'] - txt['recall@100']:+.4f}) | "
                  f"R@10 img {img['recall@10']:.4f} txt {txt['recall@10']:.4f} | "
                  f"qps {img['qps']:.0f}/{txt['qps']:.0f} B/vec {img['bytes_per_vector']} build {build_s:.1f}s",
                  flush=True)
            del index
    print(f"wrote {csv_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", nargs="+", default=["coco", "laion"])
    parser.add_argument("--variant", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument("--nbits", nargs="+", type=int, default=NBITS)
    parser.add_argument("--seeds", type=int, default=3, help=f"number of seeds, starting at common.SEED={SEED}")
    parser.add_argument("--runs", type=int, default=5, help="timed runs per query set (fastest is kept)")
    parser.add_argument("--out-dir", default="results")
    parser.add_argument("--append", action="store_true", help="append to existing CSVs instead of replacing them")
    args = parser.parse_args()

    seeds = [SEED + i for i in range(args.seeds)]
    print(f"seeds {seeds}, nbits {args.nbits}, faiss {faiss.__version__}")
    for name in args.dataset:
        data = load_dataset(name)
        print(f"\n===== {name}: corpus {data.corpus.shape}, learn_text "
              f"{None if data.learn_text is None else data.learn_text.shape} =====")
        for variant in args.variant:
            if variant == "center_qa" and data.learn_text is None:
                print(f"skip {variant} on {name}: no learn_text in the prepared file")
                continue
            sweep(data, variant, args.nbits, seeds, args.runs, args.out_dir, args.append)


if __name__ == "__main__":
    main()
