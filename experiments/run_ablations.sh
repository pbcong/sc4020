#!/usr/bin/env bash
# Runs every ablation / improvement job at once in tmux session "sc4020-abl", one window per job,
# each pinned to its own CPU block (timing uses the block's first core). Logs: results/ablations/logs/.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
LOG=results/ablations/logs
mkdir -p $LOG
SESSION=sc4020-abl
tmux kill-session -t $SESSION 2>/dev/null || true
tmux new-session -d -s $SESSION -n idle

job() {  # job <name> <cpus> <ablations.py args...>
  local name=$1 cpus=$2; shift 2
  tmux new-window -t $SESSION -n "$name" \
    "taskset -c $cpus $PY experiments/ablations.py $* 2>&1 | tee $LOG/$name.log; echo EXIT \${PIPESTATUS[0]} >> $LOG/$name.log"
}

# LAION (999k)
job l_center     0-15,128-143  centering --dataset laion
job l_qedge100k  16-23,144-151 qedges --dataset laion --n-text 100000 --tag 100k
job l_qedge300k  24-31,152-159 qedges --dataset laion --n-text 300000 --tag 300k
job l_scaling    32-39,160-167 scaling --dataset laion
job l_rerank_a   40-43,168-171 rerank --dataset laion --M 16 64 --tag a
job l_rerank_b   44-47,172-175 rerank --dataset laion --M 32 --tag b
job l_qapq_a     48-51,176-179 qapq --dataset laion --variants pq opq --tag a
job l_qapq_b     52-55,180-183 qapq --dataset laion --variants qa-text-pq qa-text-opq --tag b
job l_qapq_c     56-59,184-187 qapq --dataset laion --variants qa-mix-opq --tag c
# COCO (122k)
job c_center     64-71,192-199 centering --dataset coco
job c_hnsw_M     72-79,200-207 hnsw_params --dataset coco --configs 8,200 16,200 64,200 --tag M
job c_hnsw_efc   80-87,208-215 hnsw_params --dataset coco --configs 32,40 32,100 32,400 --tag efc
job c_rerank     88-91,216-219 rerank --dataset coco
job c_qapq       92-95,220-223 qapq --dataset coco
job c_textnodes  96-103,224-231 textnodes --dataset coco --text-frac 0.1
job c_entry      104-107,232-235 entry --dataset coco --n-text 122287
job c_qedge_full 108-111,236-239 qedges --dataset coco --n-text 122287 --tag full
job c_qedge_10pc 112-115,240-243 qedges --dataset coco --n-text 12228 --tag 10pct
job c_hnsw_rep   116-119,244-247 hnsw_params --dataset coco --configs 32,200 32,200 32,200 32,40 32,40 32,40 --tag rep
job l_qedge1m    120-127,248-255 qedges --dataset laion --n-text 999448 --tag 1m
echo "launched; attach with: tmux attach -t $SESSION"
