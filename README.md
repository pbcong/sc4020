# SC4020 Project 1: similarity search with text→image queries

## Setup
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh    # if uv is not installed
uv sync                                            # creates .venv from uv.lock
source .venv/bin/activate                          # or prefix commands with `uv run`
export SC4020_DATA=/path/to/data      # default: /raid/linzhuoyi_sc4020/data
export COCO_ROOT=/path/to/coco2014    # default: ~/coco2014
```

## Get the data
Download the prepared files:
```bash
hf download pbcong/sc4020 --repo-type dataset --local-dir $SC4020_DATA
```
Or rebuild them:
```bash
# LAION (VIBE benchmark file, 4.5 GB)
wget -P $SC4020_DATA/laion https://huggingface.co/datasets/vector-index-bench/vibe/resolve/main/laion-clip-512-normalized.hdf5
python prepare_laion.py

# COCO (needs COCO 2014 images + captions in $COCO_ROOT)
python prepare_coco.py                 # or: --device cpu --threads 64

python sanity_check.py                 # must end with ALL CHECKS PASSED
```

## Run an evaluation
```python
from evaluate import load_dataset, run_benchmark, faiss_index_bytes
import faiss, time

data = load_dataset("coco")            # or "laion"
t0 = time.time()
index = faiss.IndexPQ(data.dim, 16, 8, faiss.METRIC_INNER_PRODUCT)
index.train(data.corpus)
index.add(data.corpus)

run_benchmark(data, lambda q: index.search(q, 100)[1],
              method="PQ", build_params="M=16", search_params="-",
              build_s=time.time() - t0, index_bytes=faiss_index_bytes(index),
              csv_path="results/coco_pq.csv")
```
Each call writes two rows (image and text queries) to the CSV. Rules: see `RULES.md`.
