#!/bin/bash

DATA_DIR=$1
TASK=$2
DISP=False

echo "Training dataset... Folder: $DATA_DIR Task $TASK"

# You can parallelize these depending on how much resources you have

#############################
## Language-Conditioned Tasks
trap "kill 0" SIGINT
LANG_TASKS=$2


for task in $LANG_TASKS
    do
        # Generate data
        bash scripts/generate_gpt_datasets.sh data $task

        # TRAIN
        python cliport/train.py train.task=$task \
                        train.agent=cliport \
                        train.attn_stream_fusion_type=add \
                        train.trans_stream_fusion_type=conv \
                        train.lang_fusion_type=mult \
                        train.n_demos=200 \
                        train.n_steps=10000 \
                        train.exp_folder=exps/exps-singletask \
                        dataset.cache=True  \
                        train.batch_size=1 \
                        train.log=True

        # EVAL
        # python cliport/eval.py eval_task=$task \
        #                agent=cliport \
        #                mode=val \
        #                n_demos=100 \
        #                train_demos=100 \
        #                checkpoint_type=val_missing \
        #                exp_folder=exps 

        # TEST
        python cliport/eval.py eval_task=$task \
                       agent=cliport \
                       mode=test \
                       n_demos=100 \
                       train_demos=200 \
                       checkpoint_type=test_best \
                       exp_folder=exps/exps-singletask \
                       update_results=True \
                       disp=True
    done

python notebooks/print_results.py -r=exps-singletask

echo "Finished Training."
