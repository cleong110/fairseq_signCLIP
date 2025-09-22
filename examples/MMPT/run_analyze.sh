#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

HEIGHTS="0.0 1.1 1.2 1.3"
# HEIGHTS="1.3 1.1 1.2 0.0"

# Export HEIGHTS so GNU parallel sees it
export HEIGHTS

find results/asl_finetune_checkpoint_best/ -mindepth 4 -maxdepth 4 -type d -wholename "*cbt_033_flipped*" \
  | parallel --progress -j40 '
      for h in '"$HEIGHTS"'; do
          python analyze_scores.py {} --height_multiplier $h
          python analyze_scores.py {} --filter-eng --height_multiplier $h
      done
  '
