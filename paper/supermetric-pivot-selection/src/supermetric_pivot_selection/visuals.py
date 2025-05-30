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

# %%
conn = duckdb.connect("../../results/e_2025-05-30T15.23.47.247791.duck", read_only=True)
raw = conn.sql("select *, notes->'dataset_hash' as dataset_hash from results").pl()

merge_cols = ("seed", "dataset_supertype", "dataset_subtype", "dim")
dim_cols = ("dataset_supertype", "dataset_subtype", "dim")
measure_cols = (
    "candidate_set_size",
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

# only keep runs that have the reference column
with_optimal = raw.filter(pl.col("algorithm").str.starts_with("optimal_"))
df = raw.join(with_optimal, on=merge_cols, how="semi")

# only keep runs that have measurements for all algorithms
n_algs = df.select("algorithm").n_unique()
with_all_algs = df.group_by(merge_cols).len().filter(pl.col("len") == n_algs)
df = df.join(with_all_algs, on=merge_cols, how="semi")

# filter out failed runs
df = df.filter(pl.any_horizontal(pl.col(measure_cols) > 0))

df = df.with_columns(
    dataset=pl.col("dataset_supertype") + ", " + pl.col("dataset_subtype")
)
df = df.sort("dataset_supertype")
df

# %%
df.group_by(dim_cols).agg(
    pl.col("algorithm").gather(pl.col("candidate_set_size").arg_min()),
    pl.col("candidate_set_size").min().alias("best"),
    pl.col("candidate_set_size").max().alias("worst"),
)

# %%
hash_okay = (
    df.group_by("seed")
    .agg(pl.col("dataset").unique(), pl.col("dataset_hash").n_unique(), pl.len())
    .sort("dataset_hash")
)
hash_okay

# %%
hash_okay.group_by("dataset").agg(pl.max("dataset_hash"))

# %%
sns.histplot(
    data=df.group_by(merge_cols).agg(pl.col("dataset_hash").n_unique()),
    x="dataset_hash",
)  # hue="dataset")

# %%
df_rel = df.with_columns(
    *[
        (pl.col(col) - pl.col(col).min().over(dim_cols)).alias(col + "_zeroed")
        for col in measure_cols
    ]
).with_columns(
    *[
        (pl.col(col + "_zeroed") / pl.col(col + "_zeroed").max().over(dim_cols)).alias(
            col + "_rel"
        )
        for col in measure_cols
    ]
)

# %%
df.select("algorithm").unique().to_pandas()

# %%
sns.barplot(
    data=df.group_by(["algorithm", "dataset", "dim"]).len(),
    x="len",
    y="algorithm",
    hue="dataset",
)
plt.xlabel("datapoints per dimension")

# %%
# METRIC = "single_partition_query_share"
METRIC = "candidate_set_size_rel"

grid = sns.FacetGrid(
    data=df_rel.filter(pl.col("algorithm").is_in(important_algs)),
    col="dataset",
    col_wrap=2,
    # row="dataset_subtype",
    hue="algorithm",
    sharey=True,
    sharex=True,
    # error="ci",
)

grid.map(sns.lineplot, "dim", METRIC)


def plot_borders(**kwargs):
    data = kwargs.pop("data")
    ax = plt.gca()
    plt.grid(visible=True)
    # sns.lineplot(data=data, x="dim", y="candidate_set_size", style="algorithm")
    # ax.fill_between(data["dim"], 1, ax.get_ylim()[1], alpha=0.1, color="C0")
    # ax.fill_between(data["dim"], 1, ax.get_ylim()[0], alpha=0.1, color="C0")


grid.map_dataframe(plot_borders)
grid.add_legend()

# %%
