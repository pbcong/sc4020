#!/usr/bin/env bash
# All LSH configs at once (tmux session "sc4020-lsh"): one job per (dataset, variant, nbits), 3 seeds each,
# each pinned to its own idle physical core and spread across NUMA nodes; parts merged by merge_lsh.py.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
CORES=${CORES:-$(seq 32 127)}
LOG=results/lsh_parts/logs
mkdir -p $LOG
SESSION=sc4020-lsh
tmux kill-session -t $SESSION 2>/dev/null || true
tmux new-session -d -s $SESSION -n idle
# interleave cores across NUMA nodes (16 physical cores per node)
mapfile -t ORDER < <(for c in $CORES; do echo "$((c % 16)) $c"; done | sort -n -k1,1 -k2,2 | awk '{print $2}')
i=0
for ds in laion coco; do
  for nb in 4096 2048 1024 512 256 128 64; do
    for v in plain thr center center_qa; do
      core=${ORDER[$i]}; i=$((i + 1))
      name=${ds}_${v}_${nb}
      tmux new-window -t $SESSION -n "$name" \
        "taskset -c $core $PY -u methods/lsh.py --dataset $ds --variant $v --nbits $nb --out-dir results/lsh_parts/$name 2>&1 | tee $LOG/$name.log; echo EXIT \${PIPESTATUS[0]} >> $LOG/$name.log"
    done
  done
done
echo "launched $i jobs; attach with: tmux attach -t $SESSION"
