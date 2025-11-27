#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive
STEPS=${1-'50000'}


sh scripts/traintest_scripts/train_test_multi_task_goal_demo10.sh data \
		"[stack-block-pyramid,align-box-corner,put-block-in-bowl,packing-boxes,block-insertion,color-coordinated-sphere-insertion,rainbow-stack,vertical-insertion-blocks,stack-blocks-in-container]" \
		"[stack-block-pyramid,put-block-in-bowl,align-box-corner,packing-boxes,block-insertion]" \
		gpt5_mixcliport5_task_new 
