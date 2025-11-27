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
from sklearn.cluster import KMeans


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


def normalize_numpy_array(arr):
    return arr / (arr.max(axis=-1, keepdims=True) - arr.min(axis=-1, keepdims=True))

def fashion_scatter(
    x, class_labels, fig_name, class_names,  add_text=True
):
    # choose a color palette with seaborn.
    x = np.array(x)
    class_labels = np.array(class_labels)
    num_classes = np.max(class_labels) + 1

    # create a scatter plot.
    fig_size1, fig_size2 = 140 * 0.8, 80 * 0.6
    plt.clf()
    plt.cla()
    f = plt.figure(figsize=(fig_size1, fig_size2))
    ax = plt.subplot()

    # divide by a scale
    # x = normalize_numpy_array(x)
    for x_i in range(num_classes):
        mask = class_labels == x_i
        if mask.sum() > 0:
            sc = ax.scatter(
                x[mask, 0],
                x[mask, 1],
                lw=0,
                s=1500,
                label=class_names[x_i]
                # c=rgb_color[mask],
            )  # 40
    if add_text:
        txts = []
        for i in range(len(class_names)):
            xtext, ytext = x[i, :] # np.median(x[i, :], axis=0)
            txt = ax.text(xtext, ytext, str(class_names[i]), fontsize=40)  # 24
            txt.set_path_effects(
                [PathEffects.Stroke(linewidth=5, foreground="w"), PathEffects.Normal()]
            )
            txts.append(txt)

    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.axis("on")
    # ax.axis("tight")
    plt.savefig(fig_name +".pdf")
    plt.clf()
    print("save figure to ", fig_name)

def compute_embedding(response):
    while True:
        try:
            print('ping openai api')
            response_embedding = openai.Embedding.create(
                            model=EMBEDDING_MODEL,
                            input=response,
                        )

            response_embedding = np.array(response_embedding["data"][0]['embedding'])
            return response_embedding
        except Exception as e:
            print(e)

def draw_latent_plot(
    max_num=80,
    method="pca+tsne",
    fig_name="",
):
    # query: (response, embeddings)
    latents = []
    class_labels = []
    label_sets = []
 
    # chatgpt embedding
    total_tasks = [os.path.join("cliport/tasks", x) for x in os.listdir("cliport/tasks")] + [os.path.join("cliport/generated_tasks", x) for x in os.listdir("cliport/generated_tasks")]
    total_tasks = [t  for t in total_tasks if 'pycache' not in t and 'init' not in t \
                and 'README' not in t and 'extended' not in t and 'gripper' not in t and 'primitive' not in t\
                and 'task.py' not in t and 'camera' not in t and 'seq' not in t]
    cache_embedding_path = "output/output_embedding/task_cache_embedding.npz"
    cache_embedding = {}

    if os.path.exists(cache_embedding_path):
        cache_embedding = dict(np.load(cache_embedding_path))

    print(total_tasks) 

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
    print("latents shape:", latents.shape)
    np.savez(cache_embedding_path, **cache_embedding)

    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters, init="k-means++", random_state=42)
    kmeans.fit(latents)
    cluster_labels = kmeans.labels_

    if method == "pca+tsne":
        # reduce  dimension to the number of datapoints 
        pca = PCA(random_state=123, n_components=min(50, max_num))  # kernel PCA

        X_embedded = pca.fit_transform(latents)
        print(
            "Variance explained per principal component: {}".format(
                pca.explained_variance_ratio_[:5]
            )
        )
        print("PCA data shape:", X_embedded.shape)
        X_embedded = TSNE(random_state=123, perplexity=20).fit_transform(X_embedded)

    if method == "pca":
        pca = KernelPCA(random_state=123, n_components=2)  # kernel PCA
        X_embedded = pca.fit_transform(latents[:, :5])

    if method == "tsne":
        X_embedded = TSNE(random_state=123).fit_transform(latents)  # perplexity

    fashion_scatter(X_embedded, class_labels, fig_name, label_sets)
    fashion_scatter(X_embedded, cluster_labels, fig_name + "_cluster", label_sets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate chat-gpt embeddings")
    """
    load task descriptions from the tasks folder and embed 
    """
    parser.add_argument("--file", type=str, default="task_embedding")
    args = parser.parse_args()
    draw_latent_plot(fig_name=f'output/output_embedding/{args.file}')