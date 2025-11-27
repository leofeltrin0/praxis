#!/bin/bash

DATA_DIR=$1
TRAINTASK=${2-'[rainbow-stack,bowl-ball-placement]'}
TASKNAME=${3-'mix-two'}
STEPS=${4-'20000'}

DISP=False

echo "Training multi-task dataset... Folder: $DATA_DIR Task $TASK"
trap "kill 0" SIGINT
# You can parallelize these depending on how much resources you have

#############################
## Language-Conditioned Tasks
# [align-rope,assembling-kits-seq-seen-colors,assembling-kits-seq-unseen-colors,packing-shapes]


# TRAIN
python cliport/train.py train.task=$TRAINTASK \
                train.agent=cliport \
                train.model_task=$TASKNAME \
                train.attn_stream_fusion_type=add \
                train.trans_stream_fusion_type=conv \
                train.lang_fusion_type=mult \
                train.n_demos=200 \
                train.n_steps=${STEPS} \
                dataset.cache=True \
                train.exp_folder=exps/exp-$TASKNAME \
                dataset.type=multi  \
                train.load_from_last_ckpt=False \
                train.batchnorm=True 

# Convert Python list to Bash array
bash_array=$(python3 -c "import sys; print(' '.join((sys.argv[1])[1:-1].split(',')))" "$TRAINTASK")

# Convert the space-separated string to a bash array
echo "Testing multi-task dataset... Folder: $DATA_DIR Task $TASK"


for task in $bash_array
    do
        echo "Testing $task"
        # TEST
        bash scripts/generate_gpt_datasets.sh data $task
        
        python cliport/eval.py model_task=$TASKNAME \
                       eval_task=$task \
                       agent=cliport \
                       mode=test \
                       n_demos=100 \
                       train_demos=200 \
                       checkpoint_type=test_best \
                       type=single \
                       exp_folder=exps/exp-$TASKNAME \
                       update_results=True \
                       train.batchnorm=True  &
    done
wait

python notebooks/print_results.py -r=exps/exp-$TASKNAME
echo "Finished Training."