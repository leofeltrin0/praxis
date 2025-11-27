#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive

STEPS=${1-'15000'}
now=$(date "+%Y-%m-%d_%H-%M-%S")

sh scripts/traintest_scripts/train_test_multi_task_goal_demo10.sh data \
"[stack-block-pyramid,put-block-in-bowl]" \
"[stack-block-pyramid,put-block-in-bowl]" \
 cliport3_task_indomain_demo10_${now} $STEPS

