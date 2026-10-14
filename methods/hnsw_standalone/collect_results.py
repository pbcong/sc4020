"""collect_results.py - merge every run into one table: results/hnsw_all_results.csv

  python collect_results.py

One row per (dataset x method x efSearch x query_type):
  coco_full : Exact (brute-force baseline), HNSW (FAISS)            <- official
  coco_5k   : Exact, HNSW (FAISS), HNSW-basic (from scratch)
  laion_full, laion_5k : the same on LAION

  Time   : ms_per_query (+ _se), ms_per_query_batch, qps, build_s
  Recall : recall@1 / @10 / @100 (+ _se), delta_recall@10 (+ _se)
  Memory : index_mb (serialized index), bytes_per_vector (index bytes / n_db)
Values are means over the 1,000 queries of that type; *_se = standard error over queries.
delta_recall@10 = image - text at the same setting (same value in both rows); its SE is paired for COCO
(caption i describes image i) and unpaired for LAION (VIBE text queries are unrelated to the images).
"""
import json
import os

import pandas as pd

import config as C
from metrics import delta_se

RUNS = ["exact", "hnsw", "hnsw_basic"]
ORDER = ["dataset", "n_db", "method", "param_name", "param_value", "query_type",
         "recall@1", "recall@1_se", "recall@10", "recall@10_se", "recall@100", "recall@100_se",
         "delta_recall@10", "delta_recall@10_se",
         "ms_per_query", "ms_per_query_se", "ms_per_query_batch", "qps", "dist_comps_per_query",
         "index_mb", "bytes_per_vector", "build_s", "nodes_per_layer", "notes"]


def load(dataset, stem):
    p = lambda suffix: os.path.join(C.RAW_DIR, dataset, stem + suffix)
    if not os.path.exists(p(".csv")):
        return None
    df = pd.read_csv(p(".csv"))
    df["param_value"] = df["param_value"].fillna(-1).astype(int)       # Exact has no efSearch: -1 while merging
    pq = pd.read_csv(p("_perquery.csv"))
    pq["param_value"] = pq["param_value"].fillna(-1).astype(int)
    w = pq.pivot_table(index=["param_value", "query"], columns="query_type", values="recall@10")
    paired = C.PAIRED[C.source_of(dataset)]
    efs = w.index.get_level_values("param_value").unique()
    dl = pd.DataFrame({"delta_recall@10": [(w.loc[e, "image"] - w.loc[e, "text"]).mean() for e in efs],
                       "delta_recall@10_se": [delta_se(w.loc[e, "image"], w.loc[e, "text"], paired) for e in efs]},
                      index=efs)
    df = df.merge(dl, left_on="param_value", right_index=True, how="left")
    with open(p("_meta.json")) as fh:
        meta = json.load(fh)
    df.insert(0, "dataset", dataset)
    df["n_db"] = meta["n_db"]
    df["nodes_per_layer"] = str(meta["nodes_per_layer"]) if meta["nodes_per_layer"] else ""
    return df


TIMING_NOTES = {   # runs whose absolute times are not comparable with the others
    "laion_100k": "timed on battery power (~2.5x slower clock than the coco runs); compare times within this dataset only",
}


def note(r):
    n = [TIMING_NOTES[r["dataset"]]] if r["dataset"] in TIMING_NOTES else []
    if pd.isna(r["recall@100"]):
        n.append("recall@100 not reported: efSearch < 100 and FAISS searches with max(efSearch, k), "
                 "so a k=100 search would really run at efSearch=100")
    if r["method"] == "HNSW-basic":
        n.append("index_mb estimated as float32 vectors + int32 links (pure-Python index, not serialized)")
    if r["method"] == "Exact":
        n.append("brute-force IndexFlatIP baseline; ms_per_query_batch uses BLAS over all 1,000 queries")
    return "; ".join(n)


frames = [f for f in (load(ds, stem) for ds in C.DATASETS for stem in RUNS) if f is not None]
out = pd.concat(frames, ignore_index=True)
out["param_value"] = out["param_value"].astype("Int64").mask(out["param_value"] == -1)
out["bytes_per_vector"] = out["index_mb"] * 1e6 / out["n_db"]
out["notes"] = out.apply(note, axis=1)
out = out[ORDER]
out.to_csv(C.ALL_RESULTS, index=False)
print(f"wrote {C.ALL_RESULTS} ({len(out)} rows)")
with pd.option_context("display.width", 250):
    print(out[["dataset", "method", "param_value", "query_type", "recall@10", "recall@100",
               "delta_recall@10", "ms_per_query", "index_mb", "build_s"]].to_string(index=False))
