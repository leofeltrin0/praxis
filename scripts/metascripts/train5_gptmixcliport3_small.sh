#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive
STEPS=${1-'50000'}


sh scripts/traintest_scripts/train_test_multi_task_goal_small.sh data \
		"[put-block-in-bowl,align-box-corner,stack-block-pyramid-seq,color-coordinated-sphere-insertion,rainbow-stack,align-pair-colored-blocks-along-line,vertical-insertion-blocks,stack-blocks-in-container]" \
		"[put-block-in-bowl,align-box-corner,stack-block-pyramid-seq]" \
		gpt5_mixcliport3_task  $STEPS
