# SC4020 Project 1: similarity search with text→image queries

LSH, PQ and HNSW on CLIP embeddings (COCO, LAION), each searched with image and text queries.
The report is `report/group_XX_report.docx`.

## Layout
```
common.py, evaluate.py   shared data loading and benchmark code (imported by every script)
data_prep/               build the prepared datasets, ground truth and caption top-K caches
methods/                 one script per method: exact, lsh, pq, hnsw (+ LSH diagnostics, merge helpers)
  hnsw_standalone/       self-contained HNSW study, incl. a from-scratch implementation
experiments/             ablations and query-aware variants, plus the tmux launchers (run_*.sh)
analysis/                figures and tables for the report (compare*.py, examples.py, plot_pq.py)
results/                 CSVs, figures (results/figures), tables (results/tables), logs
report/                  report .docx and the script that builds it
```
Run every script from the repository root, e.g. `python methods/pq.py`; outputs go to `results/`.

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
python data_prep/prepare_laion.py

# COCO (needs COCO 2014 images + captions in $COCO_ROOT)
python data_prep/prepare_coco.py           # or: --device cpu --threads 64

python data_prep/sanity_check.py           # must end with ALL CHECKS PASSED
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

## Methods
```bash
python methods/exact.py --dataset coco         # brute-force baseline
python methods/lsh.py --dataset coco           # see methods/README_lsh.md
python methods/hnsw.py --dataset coco
```

## PQ
```bash
python methods/pq.py                   # sweeps M in {8,16,32,64}, nbits=8, on coco and laion
python analysis/plot_pq.py             # recall vs QPS / bytes and Δ recall plots in results/
```
`pq.py` overwrites `results/{coco,laion}_pq.csv` (use `--append` to add rows) and prints Δ recall
and reconstruction MSE per setting. Codebooks are trained on `corpus` only, k-means seed = `common.SEED` (0),
built with all cores, searched on one thread. Bits per vector = 8·M (64/128/256/512), for matching against LSH nbits.
It needs ~4 GB RAM on laion; time all methods on the same machine, plugged in, since QPS depends on CPU power state.

## Ablations, figures and report
```bash
bash experiments/run_ablations.sh      # all ablation / query-aware jobs in tmux session sc4020-abl
python analysis/compare.py             # results/figures/fig1..fig6, results/tables/
python analysis/compare_ablations.py   # results/figures/fig7..fig11, results/tables/ablations.md
python analysis/examples.py            # results/figures/fig12_examples
cd report && npm install && node build.js   # rebuilds report/group_XX_report.docx
```
