#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive
STEPS=${1-'50000'}
now=$(date "+%Y-%m-%d_%H-%M-%S")


sh scripts/traintest_scripts/train_test_multi_task_goal.sh data \
		"[stack-block-pyramid,put-block-in-bowl,color-coordinated-sphere-insertion,rainbow-stack,vertical-insertion-blocks,stack-blocks-in-container]" \
		"[stack-block-pyramid,put-block-in-bowl]" \
		train7_gpt3_mixcliport3_task_new_demo50_${now}