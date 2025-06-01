# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import sys
from glob import glob
from warnings import warn

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns

# #%config InlineBackend.figure_formats = ['svg']

# %%
merge_cols = ("seed", "dataset_supertype", "dataset_subtype", "dim")
dim_cols = ("dataset_supertype", "dataset_subtype", "dim")
measure_cols = (
    "avoided_dist_calcs",
    "useful_partition_size",
    "single_partition_query_share",
    "dummy_useful_partition_size",
    "dummy_single_partition_query_share",
)
important_algs = ("random", "optimal_candidate_set_size")
important_algs = (
    "random",
    "optimal_candidate_set_size",
    "maximize_dist",
    "remoteness",
    "IS_tri_1.5",
    "IS_pto_1.5",
)


def count_runs(msg, df):
    # count = df.n_unique(list(merge_cols) + ["algorithm"])
    count = df.group_by(
        ["algorithm", "dataset_supertype", "dataset_subtype", "dim"]
    ).agg(pl.col("seed").n_unique())
    count = count.select(pl.col("seed").min()).item()
    print(msg + f": {count}")


# %%
conn = duckdb.connect("../../results/two-days.duck", read_only=True)
raw = conn.sql("""
    SELECT
      *,
      (notes::json).dataset_hash AS dataset_hash,
      1 - candidate_set_size / 512 AS avoided_dist_calcs,
    FROM
      results
""").pl()
count_runs("total runs in dataset", raw)

raw = raw.unique(list(merge_cols) + ["algorithm"])

# only keep runs that have the reference column
with_optimal = raw.filter(pl.col("algorithm").str.starts_with("optimal_"))
df = raw.join(with_optimal, on=merge_cols, how="semi")

# only keep runs that have measurements for all algorithms
n_algs = df.select("algorithm").n_unique()
with_all_algs = df.group_by(merge_cols).len().filter(pl.col("len") == n_algs)
df = df.join(with_all_algs, on=merge_cols, how="semi")

# filter out failed runs
df = df.filter(pl.any_horizontal(pl.col(measure_cols) > 0))
count_runs("runs with complete data", df)

# df = df.filter(pl.col("useful_partition_size") > 0.001).filter(pl.col("single_partition_query_share") > 0.001)

df = df.with_columns(
    dataset=pl.col("dataset_supertype") + ", " + pl.col("dataset_subtype")
)
df = df.sort("dataset_supertype")
df.describe()

# %%
df.filter(pl.col("useful_partition_size") < 0.001).filter(
    pl.col("algorithm").str.starts_with("optim")
)

# %%
sns.countplot(
    data=df.filter(pl.col("single_partition_query_share") < 0.001),
    y="dim",
    hue="dataset",
)

# %%
(
    df.group_by(dim_cols)
    .agg(pl.col("algorithm").gather(pl.col("avoided_dist_calcs").arg_max()))
    .filter(~pl.col("algorithm").list.contains("optimal_candidate_set_size"))
)

# %%
(
    df.group_by(dim_cols)
    .agg(pl.col("algorithm").gather(pl.col("useful_partition_size").arg_max()))
    .filter(pl.col("algorithm").list.contains("optimal_hyperplane_quality"))
)

# %%
df.group_by(dim_cols).agg(
    pl.col("algorithm").gather(pl.col("candidate_set_size").arg_max()),
    pl.col("candidate_set_size").min().alias("best"),
    pl.col("candidate_set_size").max().alias("worst"),
)

# %%
# assert unique hashes
hash_not_okay = df.group_by("seed").agg(
    pl.col("dataset").unique(), pl.col("dataset_hash").n_unique(), pl.len()
)
assert hash_not_okay.filter(pl.col("dataset_hash") != 1).is_empty()

# %%
pl.col(measure_cols[0]).filter(pl.col("algorithm") == "random").over(merge_cols)

# %%
# df_rel = df.with_columns(
#    *[
#        #(pl.col(col) - pl.col(col).filter(pl.col("algorithm") == "random").mean().over(merge_cols)).alias(col+"_zeroed") #pl.col(col).min().over(merge_cols)).alias(col+"_zeroed")
#        for col in measure_cols
#     ]
# ).with_columns(
#    *[
#        (pl.col(col+"_zeroed") ).alias(col+"_rel") # / pl.col(col+"_zeroed").max().over(merge_cols)
#        for col in measure_cols
#     ]
# )

# %%
_baseline_random = (
    lambda col: pl.col(col)
    .filter(pl.col("algorithm") == "random")
    .mean()
    .over(merge_cols)
)
_best_result = lambda col: pl.col(col).max().over(merge_cols)

df_rel = df.with_columns(
    *[(pl.col(col) / _best_result(col)).alias(col + "_rel") for col in measure_cols]
).with_columns(
    *[
        ((pl.col(col) - _baseline_random(col)) / (1 - _baseline_random(col))).alias(
            col + "_mu"
        )
        for col in measure_cols
    ]
)

# %%
df

# %%
df_rel = df.with_columns(
    *[_best_result(col).alias(col + "_best") for col in measure_cols]
).join(df.filter(pl.col("algorithm") == "random"), on=merge_cols, suffix="_random")
df_rel = df_rel.with_columns(
    *[
        # (pl.col(col) - pl.col(col+"_random")) / (pl.col(col+"_best") - pl.col(col+"_random")))
        (pl.col(col) / pl.col(col + "_best")).alias(col + "_rel")
        for col in measure_cols
    ],
    *[
        (pl.col(col) - pl.col(col + "_best")).alias(col + "_abs")
        for col in measure_cols
    ],
    *[(0 - pl.col(col + "_best")).alias(col + "_abs_baseline") for col in measure_cols],
    *[
        (
            (pl.col(col) - pl.col(col + "_random"))
            / (pl.col(col + "_best") - pl.col(col + "_random"))
        ).alias(col + "_bench")
        for col in measure_cols
    ],
)

# %%
sns.barplot(
    data=df.group_by(["algorithm", "dataset", "dim"]).agg(pl.col("seed").n_unique()),
    x="seed",
    y="algorithm",
    hue="dataset",
)
plt.xlabel("datapoints per dimension")

# %%
ALGS = df.select("algorithm").unique().to_series().to_list()

# %%
plt.hist(df.select("useful_partition_size"), bins=128)
# %%
sns.histplot(
    data=df_rel,
    x="dim",
    y="useful_partition_size_rel",
    discrete=[True, False],
    bins=[20, 128],
    cbar=True,
)

# %%
algs = list(important_algs) + [
    "optimal_partition_usability",
    "optimal_hyperplane_quality",
]
data = df  # .filter(pl.col("algorithm").is_in(algs)).filter(pl.col("dataset") == "clusters, sparse")

metrics = [
    "avoided_dist_calcs_rel",
    "single_partition_query_share_rel",
    "useful_partition_size_rel",
]
# metrics = ["avoided_dist_calcs" ,"single_partition_query_share", "useful_partition_size"]

from matplotlib.colors import LogNorm

for m in metrics:
    grid = sns.FacetGrid(
        data=df_rel.filter(pl.col("algorithm").is_in(algs)),
        # row_wrap=2,
        # row="dataset",
        col="algorithm",
        sharey=True,
        sharex=True,
        # error="ci",
    )
    grid.map(
        sns.histplot,
        "dim",
        m,  # "avoided_dist_calcs",
        # bins=range(int(data["dim"].min()), int(data["dim"].max()) + 1),
        discrete=[True, False],
        bins=[20, 128 // 4],
        # pthresh = 0.05,
        cbar=True,
    )

    def plot_borders(**kwargs):
        data = kwargs.pop("data")
        ax = plt.gca()
        # set log scale even though seaborn doesn't actually support this
        # ax.collections[0].set_norm(plt.matplotlib.colors.LogNorm())
        plt.grid(visible=True)
        plt.tight_layout()
        # sns.lineplot(data=data, x="dim", y="candidate_set_size", style="algorithm")
        # ax.fill_between(data["dim"], 1, ax.get_ylim()[1], alpha=0.1, color="C0")
        # ax.fill_between(data["dim"], 1, ax.get_ylim()[0], alpha=0.1, color="C0")
        # plt.ylim(-1,1)

    grid.map_dataframe(plot_borders)


# %%
# metrics = ["avoided_dist_calcs_rel" ,"single_partition_query_share_rel", "useful_partition_size_rel"]
metrics = [
    "avoided_dist_calcs",
    "single_partition_query_share",
    "useful_partition_size",
]

for metric in metrics:
    algs = list(important_algs) + [
        "optimal_partition_usability",
        "optimal_hyperplane_quality",
    ]

    grid = sns.FacetGrid(
        data=df.filter(pl.col("algorithm").is_in(algs)),
        col="dataset",
        # row_wrap=2,
        # row="dataset_subtype",
        hue="algorithm",
        sharey=True,
        sharex=True,
        # error="ci",
    )

    grid.map(
        sns.lineplot,
        "dim",
        metric,
        estimator=np.median,
    )

    def plot_borders(**kwargs):
        data = kwargs.pop("data")
        ax = plt.gca()
        plt.grid(visible=True)
        plt.tight_layout()
        # sns.lineplot(data=data, x="dim", y="candidate_set_size", style="algorithm")
        # ax.fill_between(data["dim"], 1, ax.get_ylim()[1], alpha=0.1, color="C0")
        # ax.fill_between(data["dim"], 1, ax.get_ylim()[0], alpha=0.1, color="C0")
        # plt.ylim(-1,1)

    grid.map_dataframe(plot_borders)
    grid.add_legend()
    plt.show()

# %%
metrics = [
    "avoided_dist_calcs_rel",
    "single_partition_query_share_rel",
    "useful_partition_size_rel",
]
# metrics = ["avoided_dist_calcs" ,"single_partition_query_share", "useful_partition_size"]

for metric in metrics:
    algs = list(important_algs) + [
        "optimal_partition_usability",
        "optimal_hyperplane_quality",
    ]

    grid = sns.FacetGrid(
        data=df_rel.filter(pl.col("algorithm").is_in(algs)),
        col="dataset",
        # row_wrap=2,
        # row="dataset_subtype",
        hue="algorithm",
        sharey=True,
        sharex=True,
        # error="ci",
    )

    grid.map(
        sns.lineplot,
        "dim",
        metric,
        estimator=np.median,
    )

    def plot_borders(**kwargs):
        data = kwargs.pop("data")
        ax = plt.gca()
        plt.grid(visible=True)
        plt.tight_layout()
        # sns.lineplot(data=data, x="dim", y="candidate_set_size", style="algorithm")
        # ax.fill_between(data["dim"], 1, ax.get_ylim()[1], alpha=0.1, color="C0")
        # ax.fill_between(data["dim"], 1, ax.get_ylim()[0], alpha=0.1, color="C0")
        # plt.ylim(-1,1)

    grid.map_dataframe(plot_borders)
    grid.add_legend()
    plt.show()

# %%
metrics = [
    "avoided_dist_calcs_abs",
    "single_partition_query_share_abs",
    "useful_partition_size_abs",
]
# metrics = ["avoided_dist_calcs" ,"single_partition_query_share", "useful_partition_size"]

for metric in metrics:
    algs = list(important_algs) + [
        "optimal_partition_usability",
        "optimal_hyperplane_quality",
    ]

    grid = sns.FacetGrid(
        data=df_rel.filter(pl.col("algorithm").is_in(algs)),
        col="dataset",
        # row_wrap=2,
        # row="dataset_subtype",
        hue="algorithm",
        sharey=True,
        sharex=True,
        # error="ci",
    )

    grid.map(
        sns.lineplot,
        "dim",
        metric,
        estimator=np.median,
    )

    def plot_borders(**kwargs):
        data = kwargs.pop("data")
        ax = plt.gca()
        plt.grid(visible=True)
        plt.tight_layout()
        # sns.lineplot(data=data, x="dim", y="candidate_set_size", style="algorithm")
        # ax.fill_between(data["dim"], 1, ax.get_ylim()[1], alpha=0.1, color="C0")
        # ax.fill_between(data["dim"], 1, ax.get_ylim()[0], alpha=0.1, color="C0")
        # plt.ylim(-1,1)

    grid.map_dataframe(plot_borders)
    grid.add_legend()
    plt.show()

# %%
metrics = [
    "avoided_dist_calcs_abs",
    "single_partition_query_share_abs",
    "useful_partition_size_abs",
]
# metrics = ["avoided_dist_calcs" ,"single_partition_query_share", "useful_partition_size"]

for metric in metrics:
    algs = list(important_algs) + [
        "optimal_partition_usability",
        "optimal_hyperplane_quality",
    ]

    grid = sns.FacetGrid(
        data=df_rel.filter(pl.col("algorithm").is_in(algs)),
        col="dataset",
        # row_wrap=2,
        # row="dataset_subtype",
        hue="algorithm",
        sharey=True,
        sharex=True,
        # error="ci",
    )

    grid.map(
        sns.lineplot,
        "dim",
        metric,
        estimator=np.mean,
    )

    def plot_borders(**kwargs):
        data = kwargs.pop("data")
        ax = plt.gca()
        plt.grid(visible=True)
        plt.tight_layout()
        # sns.lineplot(data=data, x="dim", y="candidate_set_size", style="algorithm")
        # ax.fill_between(data["dim"], 1, ax.get_ylim()[1], alpha=0.1, color="C0")
        # ax.fill_between(data["dim"], 1, ax.get_ylim()[0], alpha=0.1, color="C0")
        # plt.ylim(-1,1)

    grid.map_dataframe(plot_borders)
    grid.add_legend()
    plt.show()

# %%
