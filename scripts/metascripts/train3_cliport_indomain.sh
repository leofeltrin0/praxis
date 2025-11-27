#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive

STEPS=${1-'15000'}

sh scripts/traintest_scripts/train_test_multi_task_indistribution.sh data \
"[put-block-in-bowl,align-box-corner,stack-block-pyramid-seq]" \
 cliport3_task_indomain $STEPS

