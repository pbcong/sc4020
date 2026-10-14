# Task 4 — HNSW for multimodal similarity search

How well does **HNSW** (Hierarchical Navigable Small World graph, Malkov & Yashunin 2020) find the
nearest images when the query is a **caption** instead of an **image**? The index is built once on
CLIP image vectors and searched with both query types; the gap between them (Δ recall) is the
modality-shift penalty.

## Folder
| File | Role |
|---|---|
| `config.py` | every setting: data paths, M, efConstruction, efSearch sweep, k, threads, timing |
| `prepare_data.py` | downloads the Task 1 data and writes the `.npy` arrays (data loading) |
| `hnsw_basic.py` | HNSW written from scratch from the paper's Algorithms 1–5 (`python hnsw_basic.py` = self-test) |
| `run_hnsw.py` | builds one index (exact / FAISS HNSW / from-scratch HNSW), sweeps efSearch, measures recall + time + memory |
| `metrics.py` | recall@k, standard error, per-query timing, index size |
| `plot_hnsw.py` | figures |
| `collect_results.py` | merges all runs into one table |
| `run_all.py` | reproduces everything with one command |
| `requirements.txt` | package versions used |
| `results/hnsw_all_results.csv` | **all results in one table** (time, recall, memory, standard errors) |
| `results/hnsw_plots_coco_full.png` | figures, full corpus (official) |
| `results/hnsw_plots_coco_5k.png` | figures, 5k subset, incl. from-scratch implementation |
| `results/hnsw_plots_laion_100k.png` | figures, LAION first 100,000 corpus rows (official LAION) |
| `results/hnsw_plots_laion_5k.png` | figures, LAION 5k subset, incl. from-scratch implementation |
| `results/raw/` | per-run outputs (per-query recall and time, run settings) used to build the table and figures |

## Reproduce
```bash
pip install -r requirements.txt     # Python 3.11
python run_all.py                   # ~20 min on one CPU thread; downloads 508 MB + 4.1 GB once into data/
```
`data/` is generated and not part of the submission.

> ## ⚠️ NOTE — running the FULL LAION corpus (`laion_full`, 999,448 vectors)
> Not part of `run_all.py` (which uses `laion_100k`). It needs a stronger setup and ~1–1.5 h:
>
> **Before you start**
> - Laptop **plugged in** (on battery the CPU is ~2.5× slower and times are not comparable), sleep off.
> - **≥ 4 GB free RAM** (the job uses ~2.5–3 GB) and **~9 GB free disk** (4.1 GB download + 2 GB arrays + 2.3 GB saved index).
> - Run in a **normal terminal**, not inside an IDE / AI assistant that may stop long background jobs
>   (that is what killed our earlier attempts). Close heavy apps (Zoom, browsers, databases).
> - If you already have `laion_prepared.hdf5`, copy it to `data/raw/laion_prepared.hdf5` to skip the download.
>
> **Commands** (from `hnsw/`, after `pip install -r requirements.txt`):
> ```bash
> python prepare_data.py laion_full laion_5k                         # download 4.1 GB once, write the arrays
> python run_hnsw.py --dataset laion_5k   --impl faiss                 # 1-2 min setup check (see below)
> python run_hnsw.py --dataset laion_full --impl exact                 # ~10 min, must print PIPELINE CHECK PASSED
> python run_hnsw.py --dataset laion_full --impl faiss --cache_index   # build ~35+ min, then the sweep ~10 min
> python plot_hnsw.py --dataset laion_full
> python collect_results.py
> ```
> **Setup check** — `laion_5k` must print `subset_rows sha256 91eef67c50469856423f5803f82e4745418d9618d6abc94f509903d7d8554651`
> and FAISS recall@10 (image / text) must be within 0.01 of:
> efSearch 16: 0.951 / 0.804 · 32: 0.984 / 0.902 · 64: 0.996 / 0.957 · 128: 0.998 / 0.983 · 256: 0.999 / 0.993.
> If not, stop and check the package versions (numpy must be 2.4.6 for the same random sample).
>
> **If it stops** (crash, sleep, Ctrl+C): just rerun the same `run_hnsw.py ... --cache_index` command —
> once the build has finished, the index is saved in `data/laion_full/` and is reloaded instead of rebuilt.
>
> **Send back**: `results/raw/laion_full/`, `results/hnsw_plots_laion_full.png`, `results/hnsw_all_results.csv`,
> plus your CPU model and whether the laptop was plugged in (times depend on the machine).

## Data
Task 1 output (`coco_prepared.hdf5`, MS-COCO 2014 train+val, CLIP ViT-B/32, 512-d, L2-normalised):

| | |
|---|---|
| corpus (indexed) | 122,287 image vectors |
| image queries | 1,000 images **held out** of the corpus (seed 0) |
| text queries | the first caption of the same 1,000 images |
| ground truth | exact top-100 corpus neighbours of each query (brute-force inner product) |

Recall therefore measures how well HNSW reproduces exact search, not whether the caption's own photo
is found (that photo is not in the corpus). A 5,000-vector random subset (`coco_5k`, exact top-100
recomputed) is also run so the pure-Python implementation finishes in seconds.

LAION (`laion_prepared.hdf5`, VIBE `laion-clip-512-normalized`, CLIP ViT-B/32, 512-d, L2-normalised)
has 999,448 corpus images + 1,000 held-out image queries + 1,000 VIBE text queries. Unlike COCO, text
query *i* is **unrelated** to image query *i* (so the SE of Δ is unpaired). We index the **first
100,000 corpus rows** (`laion_100k`, exact top-100 recomputed on them), which fits a 16 GB laptop and
is COCO-sized. The selection is deterministic; `python prepare_data.py laion_100k` must print
`subset_rows sha256 baa5f49fbad78af4964d9ec7eaf2d6327b2d2ca1f4dcf54e2394dfff2e36d58e`.

## Settings
- FAISS `IndexHNSWFlat(512, M=32, METRIC_INNER_PRODUCT)`, `efConstruction = 200`, **built once**.
- Sweep `efSearch ∈ {16, 32, 64, 128, 256}` on the same index; image and text queries on the same index.
- One CPU thread for building and searching.
- From-scratch HNSW uses the same M = 32 (2·M = 64 links on layer 0) and efConstruction = 200.
- Baseline: exact brute-force search (`IndexFlatIP`).

## Metrics (columns of `hnsw_all_results.csv`)
| Group | Columns | Definition |
|---|---|---|
| Recall | `recall@1`, `recall@10`, `recall@100` | mean over the 1,000 queries of \|returned top-k ∩ true top-k\| / k |
| | `delta_recall@10` | image − text at the same efSearch |
| Time | `ms_per_query` | each query searched alone (k = 10), median of 5 passes; mean over the 1,000 queries, **separately for image and text**; image and text timed interleaved so CPU drift affects both equally |
| | `ms_per_query_batch` | one call for all 1,000 queries ÷ 1,000 (cross-check) |
| | `qps` = 1000 / `ms_per_query`; `build_s` = index build time | |
| Memory | `index_mb`, `bytes_per_vector` | size of the serialized index |
| Uncertainty | `*_se` | standard error over the 1,000 queries; Δ uses the paired SE for COCO (photo *i* vs its caption *i*), unpaired for LAION |
| Cost | `dist_comps_per_query` | distance computations per query (FAISS counter) |

`recall@100` is left empty for efSearch < 100: FAISS searches with max(efSearch, k), so a k = 100
search would silently run at efSearch = 100.

## Results — COCO, full corpus (122,287 vectors), mean ± 1 SE over 1,000 queries
| efSearch | recall@10 image | recall@10 text | Δ recall@10 | ms/query image | ms/query text | dist. comp. image / text |
|---|---|---|---|---|---|---|
| 16 | 0.945 ± 0.003 | 0.716 ± 0.007 | 0.229 ± 0.007 | 0.085 | 0.087 | 457 / 496 |
| 32 | 0.981 ± 0.002 | 0.831 ± 0.005 | 0.150 ± 0.006 | 0.140 | 0.143 | 679 / 734 |
| 64 | 0.996 ± 0.001 | 0.918 ± 0.004 | 0.078 ± 0.004 | 0.245 | 0.252 | 1057 / 1158 |
| 128 | 0.999 ± 0.000 | 0.961 ± 0.002 | 0.038 ± 0.003 | 0.426 | 0.452 | 1681 / 1874 |
| 256 | 1.000 ± 0.000 | 0.986 ± 0.001 | 0.015 ± 0.001 | 0.772 | 0.808 | 2697 / 3056 |
| exact | 1 | 1 | 0 | 17.5 | 17.6 | 122,287 |

Time SEs are ≤ 0.005 ms. recall@100 (efSearch 128 / 256): image 0.992 / 0.998, text 0.912 / 0.965.

| | HNSW | Exact |
|---|---|---|
| build time | 71.7 s | 0.1 s |
| index size | 283.7 MB (2,320 B/vector) | 250.4 MB (2,048 B/vector) |
| nodes per layer | 122,287 / 3,908 / 118 / 4 / 1 | – |

## Findings
1. **Speed–recall trade-off.** Larger efSearch → more distance computations → higher recall, longer
   queries. Time roughly doubles per doubling of efSearch; there is no cliff, and it costs nothing at
   build time.
2. **Text queries lose recall, and the loss shrinks with efSearch.** Δ recall@10 falls from
   0.229 ± 0.007 (efSearch 16) to 0.015 ± 0.001 (efSearch 256) and is significant at every setting.
   The graph links image vectors to image vectors; CLIP text vectors lie in a different region of the
   space (modality gap), so the greedy search enters the graph poorly and the true neighbours are
   nearly tied. Text queries do 8–13% more distance computations and are 2–6% slower at the same
   efSearch, yet still miss more. A wider beam compensates.
3. **The penalty grows with corpus size:** at efSearch 16, Δ = 0.087 on 5k vectors vs 0.229 on 122k.
4. **Memory and build time are HNSW's costs:** it stores every full vector plus ~270 B of links per
   vector (13% more than brute force), and building takes ~72 s single-threaded.
5. **Versus exact search, it depends on how queries arrive.** One query at a time, HNSW is 22–206×
   faster (0.085–0.81 ms vs 17.5 ms). When brute force handles all 1,000 queries in one BLAS
   matrix–matrix call it costs only 0.5–0.6 ms per query: HNSW is then 6–7× faster at efSearch 16,
   only ~1.2× faster at efSearch 128, and slower at efSearch 256. At 122k vectors HNSW's advantage is
   real but modest; it grows with corpus size (brute force is linear in N, HNSW roughly logarithmic).
6. **From-scratch implementation agrees with FAISS** (5k subset): recall@10 within 0.005 at every
   efSearch, similar layer structure (5,000 / 159 / 8 vs 5,000 / 129 / 4), 6.5–8× slower (pure Python).

## Results — LAION, first 100,000 corpus rows, mean ± 1 SE over 1,000 queries
| efSearch | recall@10 image | recall@10 text | Δ recall@10 | ms/query image | ms/query text | dist. comp. image / text |
|---|---|---|---|---|---|---|
| 16 | 0.918 ± 0.005 | 0.712 ± 0.008 | 0.206 ± 0.009 | 0.199 | 0.211 | 501 / 527 |
| 32 | 0.962 ± 0.003 | 0.816 ± 0.007 | 0.146 ± 0.008 | 0.329 | 0.339 | 760 / 788 |
| 64 | 0.986 ± 0.002 | 0.895 ± 0.005 | 0.091 ± 0.005 | 0.586 | 0.604 | 1222 / 1268 |
| 128 | 0.995 ± 0.001 | 0.943 ± 0.003 | 0.052 ± 0.003 | 1.031 | 1.075 | 2026 / 2137 |
| 256 | 0.998 ± 0.001 | 0.970 ± 0.002 | 0.028 ± 0.002 | 1.937 | 2.056 | 3401 / 3700 |
| exact | 1 | 1 | 0 | 25.8 | 25.9 | 100,000 |

Time SEs are ≤ 0.012 ms. recall@100 (efSearch 128 / 256): image 0.980 / 0.993, text 0.869 / 0.930.
**These times were measured on battery power (~2.5× slower clock than the COCO run); compare them
only within this table.**

| | HNSW | Exact |
|---|---|---|
| build time | 133.0 s | 0.2 s |
| index size | 232.0 MB (2,320 B/vector) | 204.8 MB (2,048 B/vector) |
| nodes per layer | 100,000 / 3,188 / 95 / 3 | – |

LAION findings:
1. **Same shape as COCO:** Δ recall@10 falls from 0.206 ± 0.009 (efSearch 16) to 0.028 ± 0.002
   (efSearch 256) and is significant at every setting; text queries do 5–9% more distance computations.
2. **LAION text queries are harder than COCO captions at high efSearch:** at efSearch 256 text
   recall@10 is 0.970 (COCO 0.986) and recall@100 only 0.930 (COCO 0.965), although the corpus is
   slightly smaller (100k vs 122k). Image queries are also a little harder (0.918 vs 0.945 at efSearch 16).
3. **Versus exact search:** brute force over all 1,000 queries in one single-threaded BLAS call costs
   1.7–1.8 ms per query (same power state), so HNSW is 8–9× faster at efSearch 16, ~3× at 64 and
   about equal at 256.

## Caveats
- The full LAION corpus (`laion_full`) is not reported yet; see the NOTE under *Reproduce* to run it.
  `run_hnsw.py` limits BLAS/OpenMP to one thread before importing FAISS (otherwise OpenBLAS reserves
  ~3 GB for 20 threads), memory-maps `db.npy`, and `--cache_index` saves the built index.
- Absolute times depend on the machine's power/clock state: an identical earlier run on the same
  laptop was ~2× slower for everything (build 132 s). Recall is deterministic and identical across
  runs. Compare times within one run (all numbers in the table come from a single run).
- `ms_per_query` includes ~10 µs of Python→FAISS call overhead per query.
- M and efConstruction are fixed by the project spec; efConstruction affects build time and graph
  quality but not memory, and was not swept.

## Reference
Y. A. Malkov and D. A. Yashunin, "Efficient and robust approximate nearest neighbor search using
Hierarchical Navigable Small World graphs," *IEEE TPAMI* 42(4), 824–836, 2020. arXiv:1603.09320.
