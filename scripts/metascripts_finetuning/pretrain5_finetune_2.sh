#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive
STEPS=${1-'50000'}


sh scripts/traintest_scripts/train_test_multi_task_finetune_goal.sh data \
		"[color-coordinated-sphere-insertion,rainbow-stack,put-block-in-bowl,vertical-insertion-blocks,stack-blocks-in-container]" \
		"[place-red-in-green,stack-block-pyramid]" \
		gpt5_mixcliport2_finetune 
