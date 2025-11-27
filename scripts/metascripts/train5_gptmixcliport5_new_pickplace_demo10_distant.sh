#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive
STEPS=${1-'50000'}


sh scripts/traintest_scripts/train_test_multi_task_goal_demo10.sh data \
		"[stack-block-pyramid,align-box-corner,put-block-in-bowl,packing-boxes,block-insertion,manipulating-two-ropes,color-coordinated-sphere-insertion,assemble-single-car,connect-boxes-with-rope,move-piles-along-line,push-piles-into-letter]" \
		"[stack-block-pyramid,put-block-in-bowl,align-box-corner,packing-boxes,block-insertion]" \
		gpt5_mixcliport5_task_new_distant
