"""Merge the per-job LSH CSVs written by run_lsh.sh into results/<dataset>_lsh*.csv."""
import glob
import os
import re

import pandas as pd

for path in sorted(set(os.path.basename(p) for p in glob.glob("results/lsh_parts/*/*.csv"))):
    parts = sorted(glob.glob(f"results/lsh_parts/*/{path}"))
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df["_nbits"] = df["build_params"].map(lambda s: int(re.search(r"nbits=(\d+)", s).group(1)))
    df["_seed"] = df["build_params"].map(lambda s: int(re.search(r"seed=(\d+)", s).group(1)))
    df = df.sort_values(["_nbits", "_seed", "query_type"]).drop(columns=["_nbits", "_seed"])
    assert not df.duplicated(["build_params", "query_type"]).any(), f"duplicate configs in {path}"
    df.to_csv(f"results/{path}", index=False)
    print(f"results/{path}: {len(df)} rows from {len(parts)} jobs")
