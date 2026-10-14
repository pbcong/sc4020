# Benchmark rules (read before running anything)

We follow the **VIBE** protocol (Jääsaari et al., 2025, arXiv 2505.17810) and the
metric definitions of **The Faiss Library** (Douze et al., 2025).

## Datasets
| Name | Corpus (indexed) | Image queries | Text queries | Vectors |
|---|---|---|---|---|
| `laion` | ~999k LAION images (VIBE file) | 1000 held-out images | 1000 VIBE text queries | CLIP ViT-B/32, 512-d, unit norm |
| `coco`  | ~122k COCO 2014 images | 1000 held-out images | first caption of the same 1000 images | CLIP ViT-B/32, 512-d, unit norm |

Load with `data = evaluate.load_dataset("coco")`. Fields: `corpus`, `queries["image"|"text"]`,
`gt["image"|"text"]` (true top-100 ids into `corpus`), `learn_text` (extra captions).
Similarity = inner product (= cosine, vectors are unit norm).

## Rules
1. **Build on `corpus` only.** Train PQ / IVF codebooks on `corpus`. Never touch the test queries.
   `learn_text` may only be used in the clearly labelled "query-aware" improvement experiment.
2. **One index, both query types.** Build once, then search image queries and text queries on
   the same index with the same settings. Never rebuild in between. `run_benchmark` does this.
3. **Search returns top-100 ids.** Main metric `recall@100`; also report `recall@10`.
4. **VIBE timing.** `run_benchmark` sends queries one at a time, on one thread pinned to one CPU core,
   and keeps the fastest of 5 runs. Run all timed experiments on the same machine.
5. **Memory = serialized index size** (`faiss_index_bytes`), reported as bytes per vector.
6. **Build time** includes training (k-means) + adding vectors.
7. **Fix random seeds** (LSH projections, k-means) and state them.
8. **Sweep the parameter**, write every setting to your CSV in `results/`, and plot recall vs QPS.
   Never average image and text recall together: the gap between them is our main result.

## Files
- Data (not submitted): `/raid/linzhuoyi_sc4020/data/{coco,laion}/*_prepared.hdf5`
- Code: `evaluate.py` (shared, do not fork), `common.py`, `data_prep/prepare_*.py`, `data_prep/sanity_check.py`
- Results: `results/<dataset>_<method>.csv` (same columns for everyone)
