#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive

STEPS=${1-'20000'}

sh scripts/traintest_scripts/train_test_multi_task_indistribution.sh data \
"[align-rope,put-block-in-bowl,align-box-corner,stack-block-pyramid-seq,assembling-kits-seq-seen-colors]" \
 cliport5_task_indomain $STEPS

