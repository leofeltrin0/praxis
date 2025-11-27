#!/bin/bash

DATA_DIR=$1
TRAINTASK=${2-'[rainbow-stack,bowl-ball-placement]'}
STEPS=${3-'61000'}

DISP=False

echo "Training single-task dataset... Folder: $DATA_DIR Task $TRAINTASK"
trap "kill 0" SIGINT
# You can parallelize these depending on how much resources you have

#############################
## Language-Conditioned Tasks
# [align-rope,assembling-kits-seq-seen-colors,assembling-kits-seq-unseen-colors,packing-shapes]


# TRAIN
python cliport/train.py train.task=$TRAINTASK \
                train.agent=cliport \
                train.model_task=$TRAINTASK \
                train.attn_stream_fusion_type=add \
                train.trans_stream_fusion_type=conv \
                train.lang_fusion_type=mult \
                train.n_demos=200 \
                train.n_steps=${STEPS} \
                dataset.cache=True \
                train.exp_folder=exps/exp-$TRAINTASK \
                dataset.type=single  \
                train.load_from_last_ckpt=False

# Convert Python list to Bash array
bash_array=$(python3 -c "import sys; print(' '.join((sys.argv[1])[1:-1].split(',')))" "$TRAINTASK")

# # Convert the space-separated string to a bash array
# echo "Testing single-task dataset... Folder: $DATA_DIR Task $TASK"



# echo "Testing $TASK"
# # TEST
# # bash scripts/generate_gpt_datasets.sh $DATA_DIR $task

# python cliport/eval.py model_task=$TRAINTASK \
#                 eval_task=$TRAINTASK \
#                 agent=cliport \
#                 mode=test \
#                 n_demos=100 \
#                 train_demos=200 \
#                 checkpoint_type=test_best \
#                 type=single \
#                 exp_folder=exps/exp-$TRAINTASK \
#                 update_results=True 

# # wait

# python notebooks/print_results.py -r=exps/exp-$TRAINTASK/ --single
# echo "Finished Training."