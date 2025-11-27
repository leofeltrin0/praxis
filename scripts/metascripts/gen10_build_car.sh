#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive


sh scripts/traintest_scripts/train_test_multi_task_goal.sh data \
"[align-rope,sweeping-piles,align-box-corner,towers-of-hanoi-seq-seen-colors,assembling-kits-seq-seen-colors,block-insertion,palletizing-boxes,place-red-in-green,manipulating-rope,packing-boxes]" \
"[build-car]" 10taskgen_unrelated  $STEPS

sh scripts/traintest_scripts/train_test_multi_task_goal.sh data \
"[build-two-circles,build-wheel,build-bridge,towers-of-hanoi-seq-seen-colors,stack-block-pyramid-seq-seen-colors,create-pyramid-blocks-and-container,palletizing-boxes,assemble-single-car,rainbow-stack]" \
"[build-car]" 10taskgen_related  $STEPS
