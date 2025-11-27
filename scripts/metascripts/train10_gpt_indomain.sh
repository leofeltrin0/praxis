#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive

STEPS=${1-'50000'}

sh scripts/traintest_scripts/train_test_multi_task_indistribution.sh data \
		"[mix-piles,rainbow-stack,manipulating-two-ropes,insert-sphere-into-container,align-pair-colored-blocks-along-line,construct-corner-building,colorful_block-tower-on-cylinder-base,build-bridge,push_piles-into-letter]"\
		gpt10_task_indomain $STEPS
