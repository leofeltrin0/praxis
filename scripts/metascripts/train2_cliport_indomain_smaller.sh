#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive

STEPS=${1-'15000'}

sh scripts/traintest_scripts/train_test_multi_task_indistribution_smaller.sh data \
"[put-block-in-bowl,align-box-corner]" \
 cliport3_task_indomain $STEPS

