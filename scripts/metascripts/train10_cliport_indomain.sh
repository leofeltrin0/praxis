#!/bin/bash
#SBATCH -c 10
#SBATCH -n 1
#SBATCH -o logs/%j.out
#SBATCH --exclusive
STEPS=${1-'50000'}


sh scripts/traintest_scripts/train_test_multi_task_indistribution.sh data \
"[align-rope,sweeping-piles,align-box-corner,towers-of-hanoi-seq-seen-colors,assembling-kits-seq-seen-colors,block-insertion,palletizing-boxes,place-red-in-green,manipulating-rope,packing-boxes]" \
 cliport10_task_indomain $STEPS

