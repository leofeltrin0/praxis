#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive
STEPS=${1-'50000'}


sh scripts/traintest_scripts/train_test_multi_task_goal_demo10.sh data \
		"[stack-block-pyramid,put-block-in-bowl,color-coordinated-sphere-insertion,rainbow-stack,vertical-insertion-blocks,stack-blocks-in-container]" \
		"[stack-block-pyramid,put-block-in-bowl]" \
		gpt5_mixcliport2_task_new 
