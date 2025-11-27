#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive

STEPS=${1-'50000'}

sh scripts/traintest_scripts/train_test_multi_task_indistribution.sh data \
		"[sorting-blocks-into-pallets,colorful-block-tower-on-cylinder-base,align-cylinders-in-square,color-coordinated-block-tower,insert-sphere-into-container]"\
		gpt5_task_indomain_simple $STEPS
