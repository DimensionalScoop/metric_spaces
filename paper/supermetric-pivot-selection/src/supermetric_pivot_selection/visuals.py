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
from pathlib import Path

# %config InlineBackend.figure_formats = ['svg']
OUTPATH = Path("../../output/")

# %%
# renaming from code names to paper names
rename_map = {
    # dataset
    # just a wrong word
    "univariate, idd": "uniform, idd",
    "univariate, stretched": "uniform, stretched",
    # algorithms
    "maximize_dist": "max distance exact",
    "fair_max_dist": "max. p–p distance",
    "non_central_points_approx": "max. p–o variance",
    "remote_points_approx": "max. p–o distane",
    # "non_central_points": "max variance exact",
    "non_central_points_approx": "max variance",
    "different_cluster_centers": "different cluster centers",
    "gnat_dist": "max distance",
    "IS_pto_1.5": "max. o–o Ptolemy LB",
    "IS_tri_1.5": "max. o–o triangle LB",
    "random": "random",
    "optimal_candidate_set_size": "optimal $n_C$",
    "optimal_hyperplane_quality": "optimal $n_P$",
    "optimal_partition_usability": "optimal $n_U$",
    # metric names
    "avoided_dist_calcs": "avoided distance calculations (abs)",
    "useful_partition_size": "useful partition size (abs)",
    "single_partition_query_share": "queries contained in one partition (abs)",
    "avoided_dist_calcs_abs": "avoided dist. calcs.",  #  (abs. diff.)",
    "useful_partition_size_abs": "useful partition size",  #  (abs. diff.)",
    "single_partition_query_share_abs": "queries contained in one partition",  #  (abs. diff.)",
    "avoided_dist_calcs_rel": "avoided distance calculations (rel. diff.)",
    "useful_partition_size_rel": "useful partition size (rel. diff.)",
    "single_partition_query_share_rel": "queries contained in one partition (rel. diff.)",
    # unused
    "approx_Ptolemy_IS": "Ptolemy IS approx",
    "approx_cheap_Ptolemy_IS": "Ptolemy IS approx cheap",
    "approx_triangle_IS": "Triangle IS approx",
}


def translate(name_mapping_dict, *objects, ignore=("experiment_id",)):
    for obj in objects:
        if isinstance(obj, list):
            yield [
                name_mapping_dict.get(item, item) if isinstance(item, str) else item
                for item in obj
            ]
        elif isinstance(obj, str):
            yield name_mapping_dict.get(obj, obj)

        elif (
            hasattr(obj, "__class__")
            and obj.__class__.__name__ == "DataFrame"
            and hasattr(obj, "columns")
        ):
            ## Rename column names
            new_columns = {
                col: name_mapping_dict[col]
                for col in obj.columns
                if col in name_mapping_dict
            }
            df_renamed = obj.rename(new_columns)

            # Rename string values in string columns
            string_cols = [
                col
                for col in df_renamed.columns
                if df_renamed[col].dtype == pl.Utf8 and col not in ignore
            ]
            for col in string_cols:
                col_expr = pl.col(col)
                for old, new in name_mapping_dict.items():
                    col_expr = col_expr.str.replace_all(old, new, literal=True)
                df_renamed = df_renamed.with_columns(col_expr.alias(col))

            yield df_renamed


# %%
merge_cols = ("seed", "dataset_supertype", "dataset_subtype", "dim")
dim_cols = ("dataset_supertype", "dataset_subtype", "dim")
measure_cols = (
    "avoided_dist_calcs",
    "useful_partition_size",
    "single_partition_query_share",
)

optimal_algs = (
    "optimal_candidate_set_size",
    "optimal_partition_usability",
    "optimal_hyperplane_quality",
)
references_algs = ("random", *optimal_algs)

important_algs = (
    "fair_max_dist",
    "non_central_points_approx",
    "remote_points_approx",
    "IS_tri_1.5",
    "IS_pto_1.5",
)


def count_runs(msg, df):
    count = df.group_by(
        ["algorithm", "dataset_supertype", "dataset_subtype", "dim"]
    ).agg(pl.col("seed").n_unique())
    count = count.select(pl.col("seed").min()).item()
    print(msg + f": {count}")


# %%
conn = duckdb.connect("../../results/final/final-m.duck", read_only=True)
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
ALGS = df.select("algorithm").unique().to_series().to_list()
n_algs = len(ALGS)
print(f"{n_algs} algorithms")
with_all_algs = df.group_by(merge_cols).len().filter(pl.col("len") == n_algs)
df = df.join(with_all_algs, on=merge_cols, how="semi")

# filter out failed runs
df = df.filter(pl.any_horizontal(pl.col(measure_cols) > 0))
count_runs("runs with complete data", df)

df = df.with_columns(
    dataset=pl.col("dataset_supertype") + ", " + pl.col("dataset_subtype")
)
df = df.sort("dataset_supertype")

# assert unique hashes
hash_not_okay = df.group_by("seed").agg(
    pl.col("dataset").unique(), pl.col("dataset_hash").n_unique(), pl.len()
)
# assert hash_not_okay.filter(pl.col("dataset_hash") != 1).is_empty()

# %%
_best_result = lambda col: pl.col(col).max().over(merge_cols)
df_rel = df.with_columns(
    *[_best_result(col).alias(col + "_best") for col in measure_cols]
).join(df.filter(pl.col("algorithm") == "random"), on=merge_cols, suffix="_random")

df_rel = df_rel.with_columns(
    # scores relative to the best (best = 1, worst = 0)
    *[
        (pl.col(col) / pl.col(col + "_best")).alias(col + "_rel")
        for col in measure_cols
    ],
    # absolute difference to the best score (best = 0, worst = negative value)
    *[
        (pl.col(col) - pl.col(col + "_best")).alias(col + "_abs")
        for col in measure_cols
    ],
    # worst possible score for the absolute difference
    *[(0 - pl.col(col + "_best")).alias(col + "_abs_baseline") for col in measure_cols],
    # rescaled scores so that best=1 and random=0
    *[
        (
            (pl.col(col) - pl.col(col + "_random"))
            / (pl.col(col + "_best") - pl.col(col + "_random"))
        ).alias(col + "_bench")
        for col in measure_cols
    ],
)

# %%
plt.title("datapoints")
sns.barplot(
    data=df.group_by(["algorithm", "dataset", "dim"]).agg(pl.col("seed").n_unique()),
    x="seed",
    y="algorithm",
    hue="dataset",
)
plt.xlabel("datapoints per dimension")


# %%
def _per_plot_modifications(**kwargs):
    _data = kwargs.pop("data")
    ax = plt.gca()
    plt.grid(visible=True)
    plt.tight_layout()


suffix = "_abs"
# maps measure to the optimal algorithm
metrics = {
    "avoided_dist_calcs" + suffix: "optimal_candidate_set_size",
    "single_partition_query_share" + suffix: "optimal_partition_usability",
    "useful_partition_size" + suffix: "optimal_hyperplane_quality",
}

for measure, optimal_alg in metrics.items():
    use_algs = [optimal_alg] + ["random"] + list(important_algs)
    line_styles = [
        "solid",
        "solid",  # for the reference algs optimal and random
        "dotted",  # p-p
        "dashdot",  # p-o
        "dashdot",  # p-o
        "dashed",  # o-o
        "dashed",  # o-o
        # (0, (3, 1, 1, 1)),
    ] + ["dotted"] * 10
    data = df_rel.filter(pl.col("algorithm").is_in(use_algs))

    data, measure, optimal_alg, use_algs = translate(
        rename_map, data, measure, optimal_alg, use_algs
    )

    grid = sns.FacetGrid(
        data=data,
        col="dataset",
        hue="algorithm",
        hue_order=use_algs,
        hue_kws=dict(ls=line_styles),
        sharey=True,
        sharex=True,
        height=6 / 2.55,
    )

    grid.map(
        sns.lineplot,
        "dim",
        measure,
        estimator=np.mean,
        errorbar="ci",  # ("pi",0.89),
        markers=True,
    )

    grid.map_dataframe(_per_plot_modifications)
    grid.set_titles("{col_name}")
    grid.add_legend()
    plt.savefig(OUTPATH / f"{measure}.pdf")
    plt.show()

# %%
