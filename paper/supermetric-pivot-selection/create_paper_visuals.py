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

from tetrahedron import tetrahedron, proj_quality
from metric.metric import Euclid

import pivot_selection
from generate import point_generator

# %%
PATH = "paper/supermetric-pivot-selection/"
files = glob(PATH + "results/run-2/*.csv")
files += [PATH + "results/deduplicated-run-1.csv"]
df = pd.concat((pd.read_csv(f) for f in files))
df = df.drop(columns=["Unnamed: 0"])
df = df.drop_duplicates()
len(set(df.run))

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
        "dataset",
        "dim",
        "algorithm",
    ]
).apply(len, include_groups=False)
assert len(set(u)) == 1
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
def make_algos_human_readable(df):
    algo_map = dict(
        random="random",
        maximize_dist="maximize_dist",
        non_central_points="maximize_var",
        non_central_points_approx="maximize_approximated_var",
        ccs_optimal="optimal",
        hilbert_optimal="optimal",
        different_cluster_centers="different_cluster_centers",
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
    g.savefig(PATH + "fig/partition.pdf")

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
    g.savefig(PATH + "fig/css.pdf")
    return g


# %%
import patchworklib as pw

plt.clf()
pw.overwrite_axisgrid()
g0 = pw.load_seaborngrid(plot_hilbert(), label="g0")
g1 = pw.load_seaborngrid(plot_css(), label="g1")
(g1 | g0).savefig(PATH + "fig/results.pdf")
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
