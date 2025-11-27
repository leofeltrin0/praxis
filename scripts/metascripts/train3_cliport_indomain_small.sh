#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive

STEPS=${1-'15000'}
now=$(date "+%Y-%m-%d_%H-%M-%S")

sh scripts/traintest_scripts/train_test_multi_task_indistribution_small.sh data \
"[stack-block-pyramid,put-block-in-bowl,place-red-in-green]" \
 cliport3_task_indomain_demo50_2023-07-27_23-08-25 $STEPS

