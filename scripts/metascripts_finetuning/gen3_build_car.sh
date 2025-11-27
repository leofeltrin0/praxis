#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive

sh scripts/traintest_scripts/train_test_multi_task_finetune_goal.sh data "[align-rope,sweeping-piles,align-box-corner]" "[build-car]" 3taskgen_unrelated_finetune

sh scripts/traintest_scripts/train_test_multi_task_finetune_goal.sh data "[build-two-circles,build-wheel,build-bridge]" "[build-car]" 3taskgen_related_finetune
