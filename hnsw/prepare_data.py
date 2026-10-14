"""prepare_data.py - download the Task 1 data and write the .npy files used by run_hnsw.py.

  python prepare_data.py                         # every dataset in config.DATASETS
  python prepare_data.py laion_full laion_5k     # only these

For each dataset, writes data/<dataset>/:
  db.npy             (N, 512)  corpus vectors (the index is built on these)
  image_queries.npy  (1000, 512) CLIP image vectors of the held-out query images
  text_queries.npy   (1000, 512) CLIP text vectors (COCO: first caption of each query image;
                                 LAION: the 1,000 VIBE text queries, not tied to the query images)
  gt_image.npy       (1000, 100) exact top-100 corpus ids for each image query
  gt_text.npy        (1000, 100) exact top-100 corpus ids for each text query
  query_db_ids.npy   (1000,)     corpus row of each query image; all -1 (queries are held out)
  subset_rows.npy    (N,)        subsets only: which rows of the HDF5 corpus were sampled

Subsets (config.DATASETS) recompute the exact top-100 on the subset:
  *_5k        5,000 random corpus rows, np.random.default_rng(0).choice(no replacement), sorted
  laion_100k  the first 100,000 corpus rows (rows 0..99,999 of laion_prepared.hdf5)
Both are deterministic: the printed SHA-256 of subset_rows.npy must be identical on every machine.
"""
import hashlib
import os
import sys
import urllib.request

import faiss
import h5py
import numpy as np

import config as C


def download(source):
    path = C.hdf5_path(source)
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"downloading {C.SOURCES[source]} (coco 508 MB, laion 4.1 GB) ...")
    urllib.request.urlretrieve(C.SOURCES[source], path + ".tmp")
    os.replace(path + ".tmp", path)
    return path


def prepare(name):
    source, subset, how = C.DATASETS[name]
    with h5py.File(download(source), "r") as f:
        db = f["corpus"][:].astype(np.float32)
        image_q = f["queries_image"][:].astype(np.float32)
        text_q = f["queries_text"][:].astype(np.float32)
        gt_image = f["gt_image"][:].astype(np.int64)
        gt_text = f["gt_text"][:].astype(np.int64)

    rows = None
    if subset:
        if how == "first":
            rows = np.arange(subset, dtype=np.int64)
        else:
            rows = np.sort(np.random.default_rng(C.SUBSET_SEED).choice(len(db), subset, replace=False)).astype(np.int64)
        db = np.ascontiguousarray(db[rows])
        faiss.omp_set_num_threads(os.cpu_count() or 1)
        index = faiss.IndexFlatIP(db.shape[1])
        index.add(db)
        gt_image = index.search(image_q, C.GT_K)[1].astype(np.int64)
        gt_text = index.search(text_q, C.GT_K)[1].astype(np.int64)

    out = os.path.join(C.DATA_DIR, name)
    os.makedirs(out, exist_ok=True)
    arrays = {"db": db, "image_queries": image_q, "text_queries": text_q,
              "gt_image": gt_image, "gt_text": gt_text,
              "query_db_ids": np.full(len(image_q), -1, dtype=np.int64)}
    if rows is not None:
        arrays["subset_rows"] = rows
    print(f"{name}:")
    for key, a in arrays.items():
        np.save(os.path.join(out, f"{key}.npy"), a)
        print(f"  {key + '.npy':20s} {str(a.shape):14s} {a.dtype}")
    norms = np.linalg.norm(db, axis=1)
    print(f"  db norms {norms.min():.4f}-{norms.max():.4f}")
    if rows is not None:
        how_txt = "first" if how == "first" else f"random, seed {C.SUBSET_SEED}"
        print(f"  {len(rows)} of the corpus rows ({how_txt}): first {rows[:5].tolist()}, "
              f"last {rows[-3:].tolist()}")
        print(f"  subset_rows sha256 {hashlib.sha256(rows.tobytes()).hexdigest()}")


if __name__ == "__main__":
    for name in sys.argv[1:] or C.DATASETS:
        prepare(name)
