# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from uncertainties import ufloat
import seaborn as sns
from joblib import delayed, Parallel
from glob import glob
from warnings import warn
from tqdm.auto import tqdm

import sys

sys.path.append("../..")

from tetrahedron import tetrahedron, proj_quality
from metric.metric import Euclid

import pivot_selection
from generate import point_generator

# %%
# load the files with the expensive optimal results
PATH = "paper/supermetric-pivot-selection/results/"
OUT_PATH = "paper/supermetric-pivot-selection/fig/"
files = glob(PATH + "run-2/*.csv")
files += [PATH + "deduplicated-run-1.csv"]
df = pd.concat((pd.read_csv(f) for f in files))
df = df.drop(columns=["Unnamed: 0"])
df = df.drop_duplicates()
df = df.query("algorithm in ['hilbert_optimal', 'ccs_optimal']")
len(set(df.run))

# %%
# load additional files with cheaper results
files = glob(PATH + "fast-only/*.csv")
add_df = pd.concat((pd.read_csv(f) for f in files)).drop(columns=["Unnamed: 0"])

# %%
left = df
right = add_df

# only keep rows that have the same index_cols in both frames
index_cols = ["dim", "dataset", "seed", "run"]

merged = left.reset_index().merge(right.reset_index(), how="inner", on=index_cols)
keep = merged[index_cols].drop_duplicates()

left_keep = keep.merge(left, on=index_cols, how="left")
right_keep = keep.merge(right, on=index_cols, how="left")
result = pd.concat((left_keep, right_keep))

algorithms_per_sample = result.groupby(index_cols).apply(len)
assert len(
    set(algorithms_per_sample)
), "There are a different number of algorithms per sample!"

df = result

# %%
failed = df.query("note == 'failed'")
assert set(failed.dim) == {2}
assert set(failed.algorithm) == {"hilbert_optimal"}

# XXX: quick fix: let's exclude dim=2
results = df.query("dim != 2").copy()
assert len(results.query("note == 'failed'")) == 0
assert all(results.hilbert_quality >= 0)
assert all(results.hilbert_quality <= 1)
assert all(results.mean_candidate_set_size > 0)
results = results.drop(columns=["note"])

# %%
# u = df.set_index(["dataset","dim","algorithm",])
u = df.copy()
u["dub"] = u.duplicated(["hilbert_quality"], keep=False)
u.groupby("algorithm").dub.sum()
u

# %%
u = results.groupby(
    [
        "dim",
        "dataset",
        "algorithm",
    ]
).apply(len, include_groups=False)
assert (
    len(set(u)) == 1
), f"We expected to have the same number of events for each algorithm, but we got {set(u)}."
print(f"samples per (`dataset` x `dim` x `algorithm`) combination: {set(u)}")


# %%
def _normalize(df):
    # higher is better
    # lower_bound = df.query("algorithm == 'random'").hilbert_quality.mean()
    # df.hilbert_quality -= lower_bound
    upper_bound = df.query("algorithm == 'hilbert_optimal'").hilbert_quality.mean()
    df.hilbert_quality -= upper_bound
    df.hilbert_quality *= 512

    # add candidate set quality
    # best_result = df.query("algorithm == 'ccs_optimal'").mean_candidate_set_size.mean()
    # # 0 is optimal
    # df["csq"] = best_result - df.mean_candidate_set_size
    # # # 0 is random
    # df["csq"] -= df.query("algorithm == 'random'").csq.mean()
    # # 1 is optimal
    # df["csq"] /= df.query("algorithm == 'ccs_optimal'").csq.mean()
    # df = df.rename(columns=dict(csq="mean_candidate_set_quality"))

    # lower is better
    lower_bound = df.query("algorithm == 'ccs_optimal'").mean_candidate_set_size.mean()
    df.mean_candidate_set_size -= lower_bound
    # upper_bound = df.query("algorithm == 'random'").mean_candidate_set_size.mean()
    # df.mean_candidate_set_size /= upper_bound

    return df


# TODO: Decide on whether to normalize per run (grp + ["run"])
# TODO: Choose whether to do candidate set size or quality (maybe just invert the y-axis visially only?)
# TODO: Choose color scheme for beyond 1 and 0 filly

grp = ["dataset", "dim"]  # "run"]
normalized_res = results.groupby(grp).apply(_normalize, include_groups=False)
normalized_res = normalized_res.reset_index(level=grp)
normalized_res

# assert all(normalized_res.query("algorithm != 'hilbert_optimal'").hilbert_quality <= 2)
ex = normalized_res.query("dataset == 'univariate, stretched'")
sns.lineplot(ex, x="dim", y="hilbert_quality", hue="algorithm")

# %%
set(result.algorithm)


# %%
def make_algos_human_readable(df):
    algo_map = dict(
        maximize_dist="maximize dist",
        non_central_points="maximize var",
        non_central_points_approx="maximize var approx",
        approx_Ptolemy_IS="Ptolemy IS approx",
        approx_cheap_Ptolemy_IS="Ptolemy IS approx cheap",
        approx_triangle_IS="Triangle IS approx",
        different_cluster_centers="different cluster centers",
        random="random",
        ccs_optimal="optimal",
        hilbert_optimal="optimal",
    )
    num_algos = len(set(df.algorithm))
    df["algorithm"] = df.algorithm.map(algo_map)
    if num_algos != len(set(df.algorithm)):
        warn(f"discarding some algorithms entrirely!")
    return df


ex = make_algos_human_readable(normalized_res.copy())
ex = ex.query("dataset == 'univariate, stretched'")
sns.lineplot(ex, x="dim", y="hilbert_quality", hue="algorithm")

# %%
# create tables

DIM_RANGE = (5, 10)


def tabulate(value="hilbert", pivot_table=True):
    if value == "hilbert":
        quality = "hilbert_quality"
        best = np.max
    elif value == "css":
        quality = "mean_candidate_set_size"
        best = np.min
    else:
        raise NotImplementedError()

    df = normalized_res[~normalized_res.algorithm.isin([quality])]
    df = make_algos_human_readable(df)

    def error_of_mean(rows):
        return np.sqrt(np.std(rows, ddof=1) / len(rows))

    result = (
        df.query("dim >= @DIM_RANGE[0] and dim <= @DIM_RANGE[1]")
        .groupby(["dataset", "algorithm"])[quality]
        .agg(["mean", error_of_mean])
    )

    assert (
        result["error_of_mean"] < 0.5
    ).all(), "Errors to high, disable results rounding and include errors!"
    result = result.drop(columns="error_of_mean")

    result = result.drop(index="optimal", level=1)

    # result["mean"] = np.round(result["mean"],0)

    if pivot_table:
        result = result.reset_index().pivot(
            values="mean", columns="algorithm", index="dataset"
        )

        return result  # .to_latex(escape=False)
    else:
        result = result.reset_index().sort_values(["dataset", "mean"], ascending=False)
        result = result.set_index(["dataset", "algorithm"])
        return result


tabulate()


# %%
def style_pivot_table(
    df,
    output_file,
    highlighter="max",
    caption="",
):
    styler = df.style.format("{:.0f}")
    if highlighter == "max":
        styler = styler.highlight_max(props="bfseries:;", axis=1)
    elif highlighter == "min":
        styler = styler.highlight_min(props="bfseries:;", axis=1)
    else:
        raise NotImplementedError()

    # Generate LaTeX table
    latex_table = styler.to_latex(
        caption=caption,
        position="h",
        position_float="centering",
        hrules=True,
        siunitx=True,
        column_format="l" + "X" * len(styler.columns),
    )

    latex_table = (
        latex_table.replace("{tabular}", "{tabularx}")
        .replace("\\begin{tabularx}", "\\begin{tabularx}{\\textwidth}")
        .replace("{table}", "{table*}")
    )

    with open(output_file, "w") as f:
        f.write("% This is a generated file. Manual edits will be overwritten\n")
        f.write(latex_table)


s_dim_range = "\\numrange{" + f"{DIM_RANGE[0]}" + "}{" + f"{DIM_RANGE[1]}" + "}"
style_pivot_table(
    tabulate("hilbert"),
    OUT_PATH + "hilbert_table.tex",
    highlighter="max",
    caption="""Relative useful partition size, averaged over dimensions """
    + s_dim_range
    + """.
        The sizes are given as the difference to the size achievable with a optimal pivot choice on the dataset, i.e. a size of zero would be optimal.
        The best partition size for each dataset is indicated in bold.
        """,
)

style_pivot_table(
    tabulate("css"),
    OUT_PATH + "css_table.tex",
    highlighter="min",
    caption="""Relative mean candidate set size, averaged over dimensions """
    + s_dim_range
    + """.
        The sizes are given as the difference to the size achievable with a optimal pivot choice on the dataset, i.e. a size of zero would be optimal.
        The best partition size for each dataset is indicated in bold.
        """,
)


# %%

# %%
from matplotlib import rc, rcParams

# These lines are needed to get type-1 results:
# http://nerdjusttyped.blogspot.com/2010/07/type#-1-fonts-and-matplotlib-figures.html
rcParams["ps.useafm"] = True
rcParams["pdf.use14corefonts"] = True
rcParams["text.usetex"] = False

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})


# %%
def plot_hilbert():
    y = "hilbert_quality"
    y_label = "relative useful partition size"

    df = normalized_res[
        ~normalized_res.algorithm.isin(
            [
                "ccs_optimal",
                "remoteness",
                "central_and_distant",
            ]
        )
    ]
    print(set(df.algorithm))

    df = df.rename(columns={y: y_label})
    df = make_algos_human_readable(df)

    g = sns.FacetGrid(
        data=df,
        col="dataset",
        hue="algorithm",
        col_wrap=2,
        sharey=False,
    )

    def plot_borders(**kwargs):
        data = kwargs.pop("data")
        ax = plt.gca()
        plt.grid(visible=True)
        # ax.fill_between(data["dim"], 1, ax.get_ylim()[1], alpha=0.1, color="C0")
        # ax.fill_between(data["dim"], 1, ax.get_ylim()[0], alpha=0.1, color="C0")

    g.map(sns.lineplot, "dim", y_label)
    g.map_dataframe(plot_borders)

    # g.set(ylim=(-0.05, 1.1))
    g.add_legend()
    g.savefig(OUT_PATH + "partition.pdf")

    return g


# %%
def plot_css():
    y = "mean_candidate_set_size"
    y_label = "relative candidate set size"

    df = normalized_res[
        ~normalized_res.algorithm.isin(
            [
                "hilbert_optimal",
                "remoteness",
                "central_and_distant",
            ]
        )
    ].copy()

    df = df.rename(columns={y: y_label})
    df = make_algos_human_readable(df)

    g = sns.FacetGrid(
        data=df,
        col="dataset",
        hue="algorithm",
        col_wrap=2,
        sharey=False,
    )

    def plot_borders(**kwargs):
        data = kwargs.pop("data")
        ax = plt.gca()
        plt.grid(visible=True)
        # ax.fill_between(data["dim"], 1, ax.get_ylim()[1], alpha=0.1, color="C0")
        # ax.fill_between(data["dim"], 1, ax.get_ylim()[0], alpha=0.1, color="C0")

    g.map(sns.lineplot, "dim", y_label)
    g.map_dataframe(plot_borders)

    # g.set(ylim=(-0.05, 1.1))
    # g.add_legend()
    g.savefig(OUT_PATH + "css.pdf")
    return g


# %%
import patchworklib as pw

plt.clf()
pw.overwrite_axisgrid()
g0 = pw.load_seaborngrid(plot_hilbert(), label="g0")
g1 = pw.load_seaborngrid(plot_css(), label="g1")
(g1 | g0).savefig(OUT_PATH + "results.pdf")
plt.clf()

# %%
raise Execption("Unsused code")


# %%
def _better_than_random(df, score):
    rv = pd.DataFrame(index=df.algorithm.unique())
    rv.index.name = "algorithm"
    df = df.sort_values(["dataset", "dim", "run", "seed"])
    random = df.query("algorithm == 'random'")[score]

    for alg in rv.index:
        this_alg = df.query("algorithm == @alg")[score]
        better = (this_alg.to_numpy() > random.to_numpy()).sum()
        total = len(this_alg)
        rv.loc[alg, "chance_better_than_random"] = better / total

    return rv


better = results.groupby(["dim", "dataset"]).apply(
    lambda x: _better_than_random(x, "hilbert_quality")
)

better


# %%
def _is_in_top_k(df, score, k=3):
    df = df.copy()
    grp = df.groupby(["dataset", "dim", "run", "seed"])
    is_winner = grp.hilbert_quality.rank("dense", ascending=False) <= k
    df["winner"] = False
    df.loc[is_winner, "winner"] = True
    return df


exclude_optimal = ~results.algorithm.isin(["hilbert_optimal", "ccs_optimal"])
df = results[exclude_optimal]
is_winner = _is_in_top_k(df, "hilbert_quality", k=1)
wins = is_winner.groupby(["dim", "dataset", "algorithm"]).winner.sum()
samples = df.reset_index().groupby(["dim", "dataset"]).algorithm.count()
win_percent = wins / samples
win_percent.name = "win_percent"

g = sns.FacetGrid(
    data=win_percent.reset_index(),
    col="dataset",
    hue="algorithm",
    col_wrap=2,
)
g.map(sns.lineplot, "dim", "win_percent")
g.add_legend()


# %%
g = sns.FacetGrid(
    data=better.reset_index(),
    col="dataset",
    hue="algorithm",
    col_wrap=2,
)
g.map(sns.lineplot, "dim", "chance_better_than_random")
