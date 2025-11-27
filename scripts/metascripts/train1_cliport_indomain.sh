#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive

STEPS=${1-'61000'}

bash scripts/traintest_scripts/train_test_single_task_indistribution.sh data \
"place_red_in_green" \
 $STEPS

