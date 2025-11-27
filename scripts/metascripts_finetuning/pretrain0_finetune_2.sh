#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive
STEPS=${1-'50000'}


sh scripts/traintest_scripts/train_test_multi_task_finetune_goal.sh data \
		"[]" \
		"[place-red-in-green,stack-block-pyramid]" \
		gpt0_mixcliport2_finetune
