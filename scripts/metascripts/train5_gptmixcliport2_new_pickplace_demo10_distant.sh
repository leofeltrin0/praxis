#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive
STEPS=${1-'50000'}


sh scripts/traintest_scripts/train_test_multi_task_goal_demo10.sh data \
		"[stack-block-pyramid,put-block-in-bowl,manipulating-two-ropes,color-coordinated-sphere-insertion,assemble-single-car,connect-boxes-with-rope,move-piles-along-line,push-piles-into-letter]" \
		"[stack-block-pyramid,put-block-in-bowl]" \
		gpt5_mixcliport2_task_new_distant
