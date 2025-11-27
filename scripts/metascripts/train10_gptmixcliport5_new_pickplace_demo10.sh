#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive
STEPS=${1-'50000'}


sh scripts/traintest_scripts/train_test_multi_task_goal_demo10.sh data \
		"[stack-block-pyramid,align-box-corner,put-block-in-bowl,packing-boxes,block-insertion,color_linked_ball_bowl_ordering,color_specific_container_fill,insert_blocks_into_fixture,sort_insert_color_coordinated_blocks,color_ordered_blocks_on_pallet,color-coordinated-sphere-insertion,rainbow-stack,put-block-in-bowl,vertical-insertion-blocks,stack-blocks-in-container]" \
		"[stack-block-pyramid,put-block-in-bowl,align-box-corner,packing-boxes,block-insertion]" \
		gpt10_mixcliport5_task_new 
