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
N_SAMPLES = 512
# paper/supermetric-pivot-selection/
OUT_PATH = "fig/"

rng = np.random.default_rng(0xfeed2)
gens = point_generator.get_generator_dict(N_SAMPLES)
df = []
for name, func in gens.items():
    points = func(rng=rng, dim=2)
        
    df.append(dict(
        dataset=name,
    x=points[:,0],
    y=points[:,1],
    ))

df = pd.DataFrame(df).explode(["x","y"])

# %%
# TODO: refactor to give the IEEE style it's own file
from matplotlib import rc, rcParams

# These lines are needed to get type-1 results:
# http://nerdjusttyped.blogspot.com/2011/07/type#-1-fonts-and-matplotlib-figures.html
rcParams["ps.useafm"] = True
rcParams["pdf.use14corefonts"] = True
rcParams["text.usetex"] = False

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

# %%

# %%
set(df.dataset)

# %%
col_order = [
    'clusters, overlapping',
 'gaussian, circular',
 'univariate, idd',
 'clusters, sparse',
 'gaussian, eliptic',
 'univariate, stretched',
]

g = sns.FacetGrid(
    data=df,
    col="dataset",
    col_order=col_order,
    col_wrap=3,
    sharey=False,
    sharex=False,
    aspect=1,
    height=2,
)

g.map(sns.scatterplot, "x","y", marker="+")
# g.set(ylim=(-0.05, 1.1))

for ax in g.axes.flat:
    ax.set_aspect('equal', adjustable='box')
    
g.set_titles(template="{col_name}")
g.set_axis_labels(x_var="",y_var="")
g.add_legend()
g.tight_layout()
g.savefig(OUT_PATH + "datasets.pdf")

# %%
