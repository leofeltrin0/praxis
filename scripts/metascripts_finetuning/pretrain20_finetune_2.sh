#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive
STEPS=${1-'50000'}


sh scripts/traintest_scripts/train_test_multi_task_finetune_goal.sh data \
		"[color_linked_ball_bowl_ordering,color_specific_container_fill,insert_blocks_into_fixture,sort_insert_color_coordinated_blocks,color_ordered_blocks_on_pallet,color-coordinated-sphere-insertion,rainbow-stack,put-block-in-bowl,vertical-insertion-blocks,stack-blocks-in-container,'Four-corner-pyramid-challenge','create-pyramid-with-color-coded-ells','align-balls-in-colored-zones','construct-corner-blocks','color-linked-ball-bowl-ordering','create-pyramid-blocks-and-container','color-specific-container-fill','color-ordered-container-arrangement','pyramid-blocks-assemble']" \
		"[stack-block-pyramid,put-block-in-bowl,align-box-corner,packing-boxes,block-insertion]" \
		gpt20_mixcliport2_finetune 
