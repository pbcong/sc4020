import os
import time

import faiss
import h5py
import numpy as np

DATA_ROOT = os.environ.get("SC4020_DATA", "/raid/linzhuoyi_sc4020/data")
SEED = 0
N_QUERIES = 1000
GT_K = 100


def prepared_path(name):
    return os.path.join(DATA_ROOT, name, f"{name}_prepared.hdf5")


def l2_normalize(x):
    x = np.ascontiguousarray(x, dtype=np.float32)
    faiss.normalize_L2(x)
    return x


def exact_topk(corpus, queries, k=GT_K, threads=64):
    faiss.omp_set_num_threads(threads)
    index = faiss.IndexFlatIP(corpus.shape[1])
    index.add(corpus)
    t0 = time.time()
    dist, ids = index.search(queries, k)
    print(f"  exact top-{k} for {len(queries)} queries in {time.time() - t0:.1f}s")
    return dist.astype(np.float32), ids.astype(np.int64)


def write_prepared(name, arrays, attrs):
    path = prepared_path(name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with h5py.File(tmp, "w") as f:
        for key, value in arrays.items():
            f.create_dataset(key, data=value)
        for key, value in attrs.items():
            f.attrs[key] = value
    os.replace(tmp, path)
    print(f"wrote {path}")
    for key, value in arrays.items():
        print(f"  {key:16s} {str(value.shape):18s} {value.dtype}")
