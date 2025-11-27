#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive

STEPS=${1-'10000'}

sh scripts/traintest_scripts/train_test_multi_task_indistribution.sh data '[manipulating-two-ropes,construct-corner-building,color-coordinated-container-sorting,construct-corner-blocks,sort-insert-color-coordinated-blocks,insert-blocks-into-fixture,color-ordered-container-arrangement,symmetric-block-bridge-construction,connect-boxes-with-rope,vertical-insertion-blocks,cylinder-stand-alignment,insert-blocks-lineup,create-pyramid-blocks-and-container,mix-piles,multi-level-pyramid-construction,rainbow-stack,align-cylinders-in-square,align-balls-in-colored-zones,multicolor-block-bridge,align-spheres-in-colored-zones,color-blocks-in-cylinder-maze,sort-and-stack-clr-blocks,corner-block-challenge,stack-color-coordinated-blocks,assemble-single-car,color-structured-block-tower,color-sorted-block-race,sphere-align-stand,color-coordinated-block-tower,color-sorted-container-stack,color-ordered-insertion,block-pyramid-with-limited-space,sorting-blocks-into-pallets,place-ball-in-elevated-bowl,Four-corner-pyramid-challenge,color-coordinated-cylinder-tower,build-two-circles]' \
        gpt15_task_indomain