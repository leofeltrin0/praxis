#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive
STEPS=${1-'50000'}


sh scripts/traintest_scripts/train_test_multi_task_indistribution.sh data \
		"[build-wheel,build-two-circles,connect-boxes-with-rope,push-piles-into-letter,build-bridge]"\
		gpt5_task_indomain_hard $STEPS
