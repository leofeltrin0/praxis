#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive
STEPS=${1-'50000'}


sh scripts/traintest_scripts/train_test_multi_task_indistribution.sh data \
		"[mix-piles,rainbow-stack,align-pair-colored-blocks-along-line,construct-corner-blocks,stack-blocks-in-container]"\
		gpt5_task_indomain_medium $STEPS
