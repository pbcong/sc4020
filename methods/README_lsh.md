# LSH (Task 2): how to run

Three scripts, run from the repo root:

| Script | What it does | Writes |
|---|---|---|
| `methods/lsh.py` | Builds the LSH indexes and runs the benchmark sweep (nbits × seeds × variants) | `results/<dataset>_lsh*.csv` |
| `methods/lsh_tables.py` | Summary tables from those CSVs (mean ± std over seeds, Δ recall, equal-budget rows) | `results/tables/<dataset>_lsh_summary.csv` |
| `methods/lsh_diagnostic.py` | Explains the image/text recall gap of plain LSH (no timing) | `results/tables/<dataset>_lsh_diagnostic*.csv`, `.png` |

Run them in that order: the tables and the diagnostic read the sweep's CSVs.

## 1. Setup (once per terminal)

The data path must point at the folder that holds `coco/coco_prepared.hdf5`. Without it, `common.py` falls back to
the server path `/raid/...` and fails with `FileNotFoundError`.

```powershell
# PowerShell (current terminal only)
$env:SC4020_DATA = "D:/Phuc/Documents/SC4020/sc4020/data"
# or permanently, for terminals opened afterwards
setx SC4020_DATA "D:/Phuc/Documents/SC4020/sc4020/data"
```
```bash
# Linux / Git Bash
export SC4020_DATA=/path/to/data
```

## 2. Run the sweep

Full COCO sweep (4 variants × nbits 64–4096 × seeds 0, 1, 2 = 84 configs, about 25 min):

```powershell
# Windows, pinned to one core (see "Timing" below)
cmd /c "start /b /wait /affinity 10 uv run python -u methods/lsh.py --dataset coco" > results/logs/coco_lsh.log 2>&1
```
```bash
# Linux, pinned to core 0
taskset -c 0 uv run python -u methods/lsh.py --dataset coco > results/logs/coco_lsh.log 2>&1
```

Follow progress with `Get-Content results/logs/coco_lsh.log -Wait` (PowerShell) or `tail -f results/logs/coco_lsh.log`.
The run is done when the log ends with `wrote results\coco_lsh_center_qa.csv`.

### Options

| Flag | Default | Meaning |
|---|---|---|
| `--dataset` | `coco laion` | Datasets to run. **Pass `coco` explicitly**: LAION (999k vectors) takes several hours and is not part of our results yet. |
| `--variant` | all four | Any of `plain thr center center_qa` (see below) |
| `--nbits` | `64 128 256 512 1024 2048 4096` | Code length(s) to sweep |
| `--seeds` | `3` | Number of seeds, starting at `common.SEED = 0` |
| `--runs` | `5` | Timed search runs per query set; the fastest is kept |
| `--out-dir` | `results` | Where the CSVs go |
| `--append` | off | Append to existing CSVs instead of replacing them (see the warning below) |

Example, re-running just one variant (it replaces only that variant's CSV):
```powershell
cmd /c "start /b /wait /affinity 10 uv run python -u methods/lsh.py --dataset coco --variant thr"
```

### Variants

| `--variant` | CSV | Method name | What changes |
|---|---|---|---|
| `plain` | `<dataset>_lsh.csv` | `LSH` | Random hyperplanes through the origin; nothing learned. This is the main LSH result. |
| `thr` | `<dataset>_lsh_thr.csv` | `LSH-thr` | Per-bit threshold = median of the projected corpus (fitted on a seeded 100k corpus sample) |
| `center` | `<dataset>_lsh_center.csv` | `LSH-center` | Corpus (image) mean subtracted from corpus and all queries |
| `center_qa` | `<dataset>_lsh_center_qa.csv` | `LSH-center-QA` | **Query-aware**: as `center`, but text queries are centred on the text mean from `learn_text`. Report it separately and label it as query-aware (RULES.md rule 1). |

All variants are scored against the original ground truth. Each config writes two rows (image and text queries)
through `evaluate.run_benchmark`, with `build_params = "nbits=<n>;seed=<s>"`.

## 3. Summary tables and diagnostic

```powershell
uv run python methods/lsh_tables.py --dataset coco
uv run python methods/lsh_diagnostic.py --dataset coco
```

- `lsh_tables.py` prints three tables: recall by variant and nbits, the equal-budget rows (128/256/512 bits, for the
  comparison with PQ), and Δ recall (image − text) at @100 and @10. It **stops with an error** if a CSV contains the
  same config twice; re-run the sweep without `--append` to fix that.
- `lsh_diagnostic.py` (about 30 s) shows why text queries lose more recall. Panel (a) shows that each hash bit
  behaves identically for image and text pairs. Panel (b) shows that text queries' neighbours are packed about 3×
  tighter in angle, so they need about 8× more bits for the same separation. It only reads `<dataset>_lsh.csv`.

## Timing: read before running

- **Run one thing at a time.** The benchmark times searches on a single thread; anything else running alongside it
  (another sweep, the diagnostic, a large notebook) slows those searches and corrupts the QPS and build times.
  Recall is not affected.
- **Never run two sweeps into the same CSV.** `--append` alongside another run produced duplicate rows in the past.
- **Pin the process to one core** (the commands above do this), so every config is timed on the same core and
  QPS is comparable across variants and seeds. `/affinity 10` is a hex mask that selects logical CPU 4;
  `taskset -c 0` selects core 0.
- Pinning also makes the build truly single-core. Unpinned, the projection step quietly uses extra cores even
  though faiss is set to one thread.
- QPS is only comparable between runs on the **same machine**. For the cross-method plots (Task 5), every method
  must be timed on one machine.

## Reproducibility

- Projections: `faiss.RandomRotationMatrix(512, nbits)` seeded with `common.SEED + i`. `faiss.IndexLSH` hardcodes
  its own seed (5); with seed 5 our construction produces bit-identical codes to `faiss.IndexLSH(512, nbits)`.
- `thr` corpus sample and the diagnostic's random pairs: `numpy.random.default_rng(seed)`.
- Recall is deterministic for a given seed, so re-runs reproduce it exactly. Only QPS and build time vary.
- Memory (`bytes_per_vector`) is the serialized index size, so it includes the stored projection matrix
  (e.g. 580.6 B/vec at 4096 bits on COCO instead of 512 for the codes alone).
