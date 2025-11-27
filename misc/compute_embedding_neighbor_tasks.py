import torch
import torch.nn
import torchvision.models as models
from copy import deepcopy
import cv2

import cv2
import numpy as np
import sys
import itertools
import os
import IPython
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

import openai
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, KernelPCA
import seaborn as sns

import time
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import colorsys
from torchvision import datasets
import argparse
import matplotlib.patheffects as PathEffects
from scipy.spatial import cKDTree

sns.set_style("white")
sns.set_palette("muted")

font = {
    "size": 22,
}

matplotlib.rc("font", **font)
sns.set_context("paper", font_scale=3.0)


plt_param = {'legend.fontsize': 60,
             'axes.labelsize': 80,
             'axes.titlesize':80,
             'font.size'     : 80 ,
             'xtick.labelsize':80,
             'ytick.labelsize':80,
             'lines.linewidth': 10,
             'lines.color': (0,0,0)}

plt.rcParams.update(plt_param)

openai.api_key = os.getenv("OPENAI_API_KEY", "")
GPT_MODEL = "gpt4"
EMBEDDING_MODEL = "text-embedding-ada-002"
ORIGINAL_NAMES = [
    # demo conditioned
    'align-box-corner',
    'assembling-kits',
    'assembling-kits-easy',
    'block-insertion',
    'block-insertion-easy',
    'block-insertion-nofixture',
    'block-insertion-sixdof',
    'block-insertion-translation',
    'manipulating-rope',
    'packing-boxes',
    'palletizing-boxes',
    'place-red-in-green',
    'stack-block-pyramid',
    'sweeping-piles',
    'towers-of-hanoi',
    'gen-task',
    # goal conditioned
    'align-rope',
    'assembling-kits-seq',
    'assembling-kits-seq-seen-colors',
    'assembling-kits-seq-unseen-colors',
    'assembling-kits-seq-full',
    'packing-shapes',
    'packing-boxes-pairs',
    'packing-boxes-pairs-seen-colors',
    'packing-boxes-pairs-unseen-colors',
    'packing-boxes-pairs-full',
    'packing-seen-google-objects-seq',
    'packing-unseen-google-objects-seq',
    'packing-seen-google-objects-group',
    'packing-unseen-google-objects-group',
    'put-block-in-bowl',
    'put-block-in-bowl-seen-colors',
    'put-block-in-bowl-unseen-colors',
    'put-block-in-bowl-full',
    'stack-block-pyramid-seq',
    'stack-block-pyramid-seq-seen-colors',
    'stack-block-pyramid-seq-unseen-colors',
    'stack-block-pyramid-seq-full',
    'separating-piles',
    'separating-piles-seen-colors',
    'separating-piles-unseen-colors',
    'separating-piles-full',
    'towers-of-hanoi-seq',
    'towers-of-hanoi-seq-seen-colors',
    'towers-of-hanoi-seq-unseen-colors',
    'towers-of-hanoi-seq-full',
    ]


def normalize_numpy_array(arr):
    return arr / (arr.max(axis=-1, keepdims=True) - arr.min(axis=-1, keepdims=True))


def compute_embedding(response):
    for _ in range(3):
        try:
            response_embedding = openai.Embedding.create(
                            model=EMBEDDING_MODEL,
                            input=response,
                        )

            response_embedding = np.array(response_embedding["data"][0]['embedding'])
            return response_embedding
        except Exception as e:
            print(e)
            
def find_cliport_neighbor(kdtree, latents, label_sets):
    closest_embeddings, closest_idx = kdtree.query(latents, k=78)
    for i, idx in enumerate(closest_idx[0][1:]):
        s_replaced = label_sets[idx].replace("_", "-")
        if s_replaced in ORIGINAL_NAMES:
            print(label_sets[idx], i)
    

def compute_neighbors(args):  
    fig_name=f'output/output_embedding/{args.file}'
    # query: (response, embeddings)
    latents = []
    class_labels = []
    label_sets = []
 
    # chatgpt embedding
    total_tasks = [os.path.join("cliport/tasks", x) for x in os.listdir("cliport/tasks")] + [os.path.join("cliport/generated_tasks", x) for x in os.listdir("cliport/generated_tasks")]
    total_tasks = [t  for t in total_tasks if 'pycache' not in t and 'init' not in t \
                and 'README' not in t and 'extended' not in t and 'gripper' not in t and 'primitive' not in t\
                and 'task.py' not in t and 'camera' not in t and 'seq' not in t and 'seen' not in t]
    cache_embedding_path = "output/output_embedding/task_cache_embedding.npz"
    cache_embedding = {}

    if os.path.exists(cache_embedding_path):
        cache_embedding = dict(np.load(cache_embedding_path))

    # print(total_tasks) 

    for idx, task_name in enumerate(total_tasks):
        if task_name in cache_embedding:
            code_embedding = cache_embedding[task_name]
        else:
            code = open(task_name).read()
            code_embedding = compute_embedding(code)

        latents.append(code_embedding)
        label_sets.append(task_name.split("/")[-1][:-3])
        cache_embedding[task_name] = code_embedding
        class_labels.append(idx)

    latents = np.array(latents)
    # print("latents shape:", latents.shape)
    # np.savez(cache_embedding_path, **cache_embedding)

    target_task_idx = label_sets.index(args.target_task)
    kdtree = cKDTree(latents)
    closest_embeddings, closest_idx = kdtree.query(latents[[target_task_idx]], k=args.num+1)
    # print(latents.shape, args.num, target_task_idx, closest_idx,label_sets)
   
    print(f"closest tasks to {args.target_task}: {[label_sets[task] for task in closest_idx[0][1:]]}")
    
    # print(f"closest tasks in cliport original tasks: {find_cliport_neighbor(kdtree, latents[[target_task_idx]], label_sets)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate chat-gpt embeddings")
    """
    load task descriptions from the tasks folder and embed 
    """
    parser.add_argument("--file", type=str, default="task_embedding")
    parser.add_argument("--target_task", type=str, default="align_box_corner")
    parser.add_argument("--num", type=int, default=3)

    args = parser.parse_args()
    compute_neighbors(args)