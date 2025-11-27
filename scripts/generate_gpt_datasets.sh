#!/bin/bash

DATA_DIR=/home/yzc/shared/project/GPT-CLIPort/data
TASK='put-block-in-bowl align-box-corner stack-block-pyramid-seq align-pair-colored-blocks-along-line vertical-insertion-blocks stack-blocks-in-container'
DISP=False

echo "Generating dataset... Folder: $DATA_DIR"

# sh scripts/generate_gpt_datasets.sh data "align-rope assembling-kits-seq-seen-colors assembling-kits-seq-unseen-colors packing-shapes packing-boxes-pairs-seen-colors packing-boxes-pairs-unseen-colors packing-seen-google-objects-seq packing-unseen-google-objects-seq packing-seen-google-objects-group packing-unseen-google-objects-group put-block-in-bowl-seen-colors put-block-in-bowl-unseen-colors stack-block-pyramid-seq-seen-colors stack-block-pyramid-seq-unseen-colors separating-piles-seen-colors separating-piles-unseen-colors towers-of-hanoi-seq-seen-colors towers-of-hanoi-seq-unseen-colors
# sh scripts/generate_gpt_datasets.sh data "assemble-single-car stack-color-coordinated-blocks color-structured-block-tower insert-blocks-into-fixture construct-corner-building colored-cylinder-in-square color-coordinated-block-tower build-house align-pair-colored-blocks-along-line insert-sphere-into-container build-wheel build-two-circles build-car build-bridge manipulating-two-ropes rainbow-stack mix-piles stack-blocks-in-container" 
# You can parallelize these depending on how much resources you have

#############################
## Language-Conditioned Tasks

# LANG_TASKS='align-rope assembling-kits-seq-seen-colors'
# trap "kill 0" SIGINT

# LANG_TASKS='place_red_in_green'
LANG_TASKS='rainbow-stack'


for task in $LANG_TASKS
    do
        python cliport/demos.py n=200  task=$task mode=train data_dir=$DATA_DIR disp=$DISP &
        python cliport/demos.py n=50   task=$task mode=val   data_dir=$DATA_DIR disp=$DISP &
        python cliport/demos.py n=100  task=$task mode=test  data_dir=$DATA_DIR disp=$DISP &
    done
wait

echo "Finished Language Tasks."


