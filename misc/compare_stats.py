import matplotlib as mpl

mpl.use("Agg")
import argparse
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import IPython

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
    mkdir_if_missing(f"output/output_figures/{name[:30]}")
    plt.savefig(f"output/output_figures/{name[:30]}/output.png")
    plt.clf()


def main(multirun_out, title):
    dfs = []
    suffix = ""
    run_num = 0

    for rundir in (sorted(multirun_out.split(","))):
        runpath = os.path.join('output/output_stats', rundir)
        statspath = os.path.join(runpath, "eval_results.csv")
        if os.path.exists(statspath):
            run_num += 1
            df = pd.read_csv(statspath)
            dfs.append(df)

    # merge dfs, which have shared column names
    df = pd.concat(dfs)
    title += f" run: {run_num} "

    # rewards
    fig, ax = plt.subplots(figsize=(16, 8))
    sns_plot = sns.barplot(
        data=df, x="task", y="success", hue='method', errorbar=("sd", 1), palette="deep"
    )

    # label texts
    for container in ax.containers:
        ax.bar_label(container, label_type="center", fontsize="x-large", fmt="%.2f")
    ax.set_xticklabels(['\n'.join(str(xlabel.get_text()).split("-")) for xlabel in ax.get_xticklabels()])

    # save plot
    save_figure(f"{multirun_out}_{title}{suffix}", title)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--multirun_out", type=str)
    parser.add_argument("--title", type=str, default="")

    args = parser.parse_args()
    main(args.multirun_out, args.title)
