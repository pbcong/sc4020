"""run_all.py - reproduce every HNSW result with one command (single thread; COCO ~10 min, LAION ~10 min).

  python run_all.py

  1. download the Task 1 data and write data/<dataset>/*.npy          (prepare_data.py)
  2. self-test the from-scratch HNSW on random data                     (hnsw_basic.py)
  then for COCO and for LAION:
  3. exact search on each dataset: pipeline check (recall must be 1.0) + baseline
  4. FAISS HNSW on the main set (coco_full, laion_100k = first 100k LAION rows) and <source>_5k;
     from-scratch HNSW on <source>_5k
  5. figures; finally the single results table results/hnsw_all_results.csv
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

STEPS = [["prepare_data.py"], ["hnsw_basic.py"]]
MAIN = {"coco": "coco_full", "laion": "laion_100k"}   # laion_full (999k) needs ~35 min and >2.5 GB free RAM
for source, full in MAIN.items():
    small = f"{source}_5k"
    STEPS += [
        ["run_hnsw.py", "--dataset", full, "--impl", "exact"],
        ["run_hnsw.py", "--dataset", small, "--impl", "exact"],
        ["run_hnsw.py", "--dataset", full, "--impl", "faiss", "--cache_index"],
        ["run_hnsw.py", "--dataset", small, "--impl", "faiss"],
        ["run_hnsw.py", "--dataset", small, "--impl", "basic"],
        ["plot_hnsw.py", "--dataset", full],
        ["plot_hnsw.py", "--dataset", small],
    ]
STEPS.append(["collect_results.py"])

for step in STEPS:
    print(f"\n===== python {' '.join(step)}", flush=True)
    subprocess.run([sys.executable, "-u", *step], check=True, cwd=HERE)
print("\nALL DONE - see results/")
