#!/usr/bin/env bash
# Runs every PQ / HNSW / exact job at once in tmux session "sc4020", one window per job.
# Each job is pinned to its own CPU block (taskset); builds use the whole block, timing uses
# the block's first core only (evaluate.timed_search pins to min(affinity)).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
mkdir -p results/logs results/parts
SESSION=sc4020
tmux kill-session -t $SESSION 2>/dev/null || true
tmux new-session -d -s $SESSION -n idle

job() {  # job <name> <cpus> <command...>
  local name=$1 cpus=$2; shift 2
  tmux new-window -t $SESSION -n "$name" \
    "taskset -c $cpus $* 2>&1 | tee results/logs/$name.log; echo EXIT \${PIPESTATUS[0]} >> results/logs/$name.log"
}

job laion_hnsw  32-47,160-175 $PY hnsw.py --dataset laion
job laion_exact 48-49         $PY exact.py --dataset laion
job laion_pq64  50-53,178-181 $PY pq.py --datasets laion --M 64 --build-threads 8 --out results/parts/laion_pq_M64.csv
job laion_pq32  54-57,182-185 $PY pq.py --datasets laion --M 32 --build-threads 8 --out results/parts/laion_pq_M32.csv
job laion_pq16  58-60,186-188 $PY pq.py --datasets laion --M 16 --build-threads 6 --out results/parts/laion_pq_M16.csv
job laion_pq8   61-63,189-191 $PY pq.py --datasets laion --M 8  --build-threads 6 --out results/parts/laion_pq_M8.csv
job coco_hnsw   64-71,192-199 $PY hnsw.py --dataset coco
job coco_pq     72-75,200-203 $PY pq.py --datasets coco --build-threads 8
job coco_exact  76-77         $PY exact.py --dataset coco
echo "launched; attach with: tmux attach -t $SESSION"
