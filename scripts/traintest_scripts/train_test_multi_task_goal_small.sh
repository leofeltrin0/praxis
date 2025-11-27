#!/bin/bash

DATA_DIR=$1
TRAINTASK=${2-'[rainbow-stack,bowl-ball-placement]'}
TESTTASK=${3-'[rainbow-stack,bowl-ball-placement]'}
TASKNAME=${4-'mix-two'}
STEPS=${5-'10000'}

DISP=False

echo "Training multi-task dataset... Folder: $DATA_DIR Task $TRAINTASK"

# You can parallelize these depending on how much resources you have

#############################
## Language-Conditioned Tasks
# [align-rope,assembling-kits-seq-seen-colors,assembling-kits-seq-unseen-colors,packing-shapes,stack-block-pyramid-seq-unseen-colors,
# separating-piles-seen-colors,separating-piles-unseen-colors,towers-of-hanoi-seq-seen-colors,towers-of-hanoi-seq-unseen-colors]

# example: sh scripts/traintest_scripts/train_test_multi_task_indistribution.sh data "[align-rope,sweeping-piles,align-box-corner,block-insertion,manipulating-rope,place-red-in-green]" 6taskindomain
# sh scripts/traintest_scripts/train_test_multi_task_goal.sh data "[align-rope,sweeping-piles,align-box-corner,block-insertion,manipulating-rope,place-red-in-green]" "[towers-of-hanoi]" 6taskgen
# sh scripts/traintest_scripts/train_test_multi_task_goal.sh data "[align-rope,sweeping-piles,align-box-corner]" "[towers-of-hanoi]" 3taskgen
# sh scripts/traintest_scripts/train_test_multi_task_goal.sh data "[align-rope]" "[towers-of-hanoi]" 1taskgen
# sh scripts/traintest_scripts/train_test_multi_task_goal.sh data "[align-rope,sweeping-piles,align-box-corner,block-insertion,manipulating-rope,place-red-in-green]" "[towers-of-hanoi]" 10taskgen

trap "kill 0" SIGINT
# increase the nums of epoch
python cliport/train.py train.task=$TRAINTASK \
                train.agent=cliport \
                train.model_task=$TASKNAME \
                train.attn_stream_fusion_type=add \
                train.trans_stream_fusion_type=conv \
                train.lang_fusion_type=mult \
                train.n_demos=50 \
                train.n_steps=${STEPS} \
                dataset.cache=True \
                train.exp_folder=exps/exp-$TASKNAME-small  \
                dataset.type=multi   \
                train.load_from_last_ckpt=False


# Convert Python list to Bash array
bash_array=$(python3 -c "import sys; print(' '.join((sys.argv[1])[1:-1].split(',')))" "$TESTTASK")

# Convert the space-separated string to a bash array
echo "Testing multi-task dataset... Folder: $DATA_DIR Task $TESTTASK"


for task in $bash_array
    do
        echo "Testing $task"
        # TEST
        # bash scripts/generate_gpt_datasets.sh data $task
        
        python cliport/eval.py model_task=$TASKNAME \
                       eval_task=$task \
                       agent=cliport \
                       mode=test \
                       n_demos=100 \
                       train_demos=50 \
                       checkpoint_type=test_best \
                       type=single \
                       exp_folder=exps/exp-$TASKNAME-small   \
                       update_results=True &
    done
wait

python notebooks/print_results.py -r=exps/exp-$TASKNAME-small

echo "Finished Training."