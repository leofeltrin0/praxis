import os
import sys
import json

from cliport import agents
from cliport import tasks
import argparse
import datetime
import matplotlib as mpl

mpl.use("Agg")
import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import IPython
import numpy as np
font = {
    "size": 22,
}
matplotlib.rc("font", **font)
sns.set_context("paper", font_scale=2.0)


def mkdir_if_missing(dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)


def save_figure(name, title=""):
    print(f"output/output_figures/{name}.png")
    if len(title) > 0:
        plt.title(title)
    plt.tight_layout()
    mkdir_if_missing(f"output/output_figures/{name}")
    plt.savefig(f"output/output_figures/{name}/output.png")
    plt.clf()


def print_and_write(file_handle, text):
    print(text)
    if file_handle is not None:
        file_handle.write(text + "\n")
    return text

parser = argparse.ArgumentParser()

# federated arguments (Notation for the arguments followed from paper)
parser.add_argument(
    "--results", "-r", type=str, default="exps/exps-singletask"
)
parser.add_argument(
    "--single", "-s", action="store_true", default=False
)
args = parser.parse_args()

root_folder = os.environ['GENSIM_ROOT']
exp_folder = os.path.join(root_folder, args.results) # replace 'cliport_quickstart' with your exps folder


mkdir_if_missing('output/output_figures')
mkdir_if_missing('output/cliport_output')
mkdir_if_missing('output/output_stat')



output_stat_file = os.path.join('output/', 'cliport_output/', 'cliport-training.txt')
mkdir_if_missing('output/cliport_output/')
file_handle = open(output_stat_file, 'a+')

tasks_list = list(tasks.names.keys())
agents_list = list(agents.names.keys())
demos_list = [1, 5, 10, 20, 30, 50, 100, 200, 1000] # 100,

results = {}
for t in tasks_list:
    for a in agents_list:
        for d in demos_list:
            task_folder = f'{t}-{a}-n{d}-train'
            task_folder_path = os.path.join(exp_folder, task_folder, 'checkpoints')

            if os.path.exists(task_folder_path):
                print(f"train {task_folder_path}")

                jsons = [f for f in os.listdir(task_folder_path) if '.json' in f]
                for j in jsons:
                    model_type = 'multi' if 'multi' in j else 'single'
                    eval_type = 'val' if 'val' in j else 'test'
                    
                    with open(os.path.join(task_folder_path, j)) as f:
                        res = json.load(f)
                    
                    results[f'{t}-{a}-n{d}-{model_type}-{eval_type}'] = res

dt_string = datetime.datetime.now().strftime("%d_%m_%Y_%H:%M:%S")
print_and_write(file_handle, f"==========================={dt_string}=========================\n")
print_and_write(file_handle, f'Experiments folder: {exp_folder}\n')

data = {'task': [], 'success': []}

for eval_type in ['val', 'test']:
    print_and_write(file_handle, f'----- {eval_type.upper()} -----\n')
    for t in tasks_list:
        for a in agents_list:
            for d in demos_list:
                for model_type in ['single', 'multi']:
                    eval_key = f'{t}-{a}-n{d}-{model_type}-{eval_type}'
                    
                    if eval_key in results:    
                        print_and_write(file_handle, f'{eval_key} {t} | Train Demos: {d}')
                        res = results[eval_key]
                        best_score, best_ckpt = max(zip([v['mean_reward'] for v in list(res.values())], res.keys()))
                        # TODO: test that this works for full results folder
                        
                        print_and_write(file_handle, f'\t{best_score*100:1.1f} : {a} | {model_type}\n')
                        data['task'].append(t)
                        data['success'].append(best_score)

data['task'].append("Average")
data['success'].append(np.mean(data["success"]))


# make figure as well for sinle expeirment results
dfs = []
suffix = ""
run_num = 0
df = pd.DataFrame.from_dict(data)
title = args.results + "_res"

# rewards
fig, ax = plt.subplots(figsize=(16, 8))
sns_plot = sns.barplot(
    data=df, x="task", y="success", errorbar=("sd", 1), palette="deep"
)

# label texts
for container in ax.containers:
    ax.bar_label(container, label_type="center", fontsize="x-large", fmt="%.2f")
# ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right")
ax.set_xticklabels(['\n'.join(str(xlabel.get_text()).split("-")) for xlabel in ax.get_xticklabels()])

# save plot
save_figure(f"{title}", title)
