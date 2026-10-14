"""Summary tables for the LSH sweeps: mean +- std over seeds, image and text kept separate.

delta = recall(image queries) - recall(text queries), computed per seed on the same index, then aggregated.
"""
import argparse
import os

import pandas as pd

SUFFIXES = ["lsh", "lsh_thr", "lsh_center", "lsh_center_qa"]
EQUAL_BUDGET_BITS = [128, 256, 512]


def load(dataset, results_dir, suffixes=SUFFIXES):
    frames = []
    for suffix in suffixes:
        path = os.path.join(results_dir, f"{dataset}_{suffix}.csv")
        if os.path.exists(path):
            frames.append(pd.read_csv(path))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    # a repeated config (e.g. from a concurrent --append run) would otherwise be averaged silently by pivot_table
    keys = ["method", "build_params", "query_type"]
    dup = df[df.duplicated(keys, keep=False)]
    if len(dup):
        configs = sorted(set(map(tuple, dup[keys[:2]].values)))
        raise ValueError(f"{dataset}: {len(dup)} rows repeat a (method, build_params, query_type); "
                         f"re-run without --append. Duplicated configs: {configs}")
    params = df["build_params"].str.extract(r"nbits=(\d+);seed=(\d+)").astype(int)
    df["nbits"], df["seed"] = params[0], params[1]
    return df


def per_seed(df):
    """One row per (method, nbits, seed) with image and text side by side."""
    keys = ["method", "nbits", "seed"]
    wide = df.pivot_table(index=keys, columns="query_type", values=["recall@100", "recall@10", "qps"])
    wide.columns = [f"{m}_{q}" for m, q in wide.columns]
    shared = df.groupby(keys)[["bytes_per_vector", "index_mb", "build_s"]].first()
    out = wide.join(shared).reset_index()
    for k in ("100", "10"):
        out[f"delta@{k}"] = out[f"recall@{k}_image"] - out[f"recall@{k}_text"]
    return out


def fmt(mean, std):
    return f"{mean:.4f}+-{std:.4f}"


def summary(seeds):
    cols = ["recall@100_image", "recall@100_text", "delta@100", "recall@10_image", "recall@10_text", "delta@10"]
    g = seeds.groupby(["method", "nbits"])
    mean, std = g[cols].mean(), g[cols].std(ddof=0).fillna(0.0)
    table = pd.DataFrame({c: [fmt(m, s) for m, s in zip(mean[c], std[c])] for c in cols}, index=mean.index)
    other = g.agg(qps_image=("qps_image", "mean"), bytes_per_vector=("bytes_per_vector", "first"),
                  build_s=("build_s", "mean"), n_seeds=("seed", "nunique"))
    return table.join(other.round({"qps_image": 0, "build_s": 2}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", nargs="+", default=["coco", "laion"])
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)

    for name in args.dataset:
        df = load(name, args.results_dir)
        if df is None:
            print(f"no LSH results for {name} in {args.results_dir}")
            continue
        table = summary(per_seed(df))
        print(f"\n===== {name}: recall by method and nbits (mean+-std over seeds) =====")
        print(table.to_string())

        print(f"\n===== {name}: equal-budget rows for Table B (bits per vector = nbits) =====")
        eq = table.loc[table.index.get_level_values("nbits").isin(EQUAL_BUDGET_BITS)]
        print(eq[["recall@100_image", "recall@100_text", "recall@10_image", "recall@10_text", "bytes_per_vector"]]
              .to_string())

        print(f"\n===== {name}: modality shift, delta@100 and delta@10 per variant =====")
        print(table[["delta@100", "delta@10"]].unstack("method").to_string())

        # kept out of results/*.csv so a glob-merge of benchmark CSVs does not pick it up
        out = os.path.join(args.results_dir, "tables", f"{name}_lsh_summary.csv")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        table.to_csv(out)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
