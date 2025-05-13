"""Examplatory 2D visualizations of the dataset generators"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

from generate import point_generator

N_SAMPLES = 512
OUT_PATH = "output/"

rng = np.random.default_rng(0xFEED2)
gens = point_generator.get_generator_dict(N_SAMPLES)
distributions = []
for name, func in gens.items():
    points = func(rng=rng, dim=2)

    distributions.append(
        dict(
            dataset=name,
            x=points[:, 0],
            y=points[:, 1],
        )
    )

distributions = pd.DataFrame(distributions).explode(["x", "y"])


# These lines are needed to get type-1 results:
# http://nerdjusttyped.blogspot.com/2011/07/type#-1-fonts-and-matplotlib-figures.html
rcParams["ps.useafm"] = True
rcParams["pdf.use14corefonts"] = True
rcParams["text.usetex"] = False

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

distributions = distributions.dataset.replace(
    {"univariate, idd": "uniform, idd", "univariate, stretched": "uniform, stretched"},
)

col_order = [
    "clusters, overlapping",
    "gaussian, circular",
    "uniform, idd",
    "clusters, sparse",
    "gaussian, eliptic",
    "uniform, stretched",
]


g = sns.FacetGrid(
    data=distributions,
    col="dataset",
    col_order=col_order,
    col_wrap=3,
    sharey=False,
    sharex=False,
    aspect=1,
    height=2,
)

g.map(sns.scatterplot, "x", "y", marker="+")

for ax in g.axes.flat:
    ax.set_aspect("equal", adjustable="box")

g.set_titles(template="{col_name}")
g.set_axis_labels(x_var="", y_var="")
g.add_legend()
g.tight_layout()
g.savefig(OUT_PATH + "datasets.pdf")
