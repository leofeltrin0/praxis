#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive
STEPS=${1-'50000'}


sh scripts/traintest_scripts/train_test_multi_task_goal_demo10.sh data \
		"[stack-block-pyramid,color-coordinated-sphere-insertion,rainbow-stack,put-block-in-bowl,vertical-insertion-blocks]" \
		"[put-block-in-bowl,stack-block-pyramid]" \
		gpt3_mixcliport2_task_new 
