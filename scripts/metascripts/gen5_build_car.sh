#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive

STEPS=${1-'15000'}

sh scripts/traintest_scripts/train_test_multi_task_goal.sh data \
	"[align-rope,sweeping-piles,align-box-corner,towers-of-hanoi-seq-seen-colors,assembling-kits-seq-seen-colors]" "[build-car]" \
	 5taskgen_unrelated $STEPS

sh scripts/traintest_scripts/train_test_multi_task_goal.sh data \
"[build-two-circles,build-wheel,build-bridge,towers-of-hanoi-seq-seen-colors,stack-block-pyramid-seq-seen-colors]" "[build-car]" \
     5taskgen_related  $STEPS
