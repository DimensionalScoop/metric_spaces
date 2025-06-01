"""Examplatory 2D visualizations of the dataset generators"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rcParams

from meters.generate.point_generator import GENERATORS

N_SAMPLES = 512
OUT_PATH = "output/"

rng = np.random.default_rng(0xFEED2)
distributions = []
for name, func in GENERATORS.items():
    points = func(rng=rng, dim=2, n_samples=N_SAMPLES)
    if name in ("gaussian, eliptic", "univariate, stretched"):
        points = points[:, ::-1]

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
# rcParams["ps.useafm"] = True
# rcParams["pdf.use14corefonts"] = True
# rcParams["text.usetex"] = False

# plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})

distributions["dataset"] = distributions.dataset.replace(
    {"univariate, idd": "uniform, idd", "univariate, stretched": "uniform, stretched"},
)

col_order = [
    "clusters, overlapping",
    "clusters, sparse",
    "gaussian, circular",
    "gaussian, eliptic",
    "uniform, idd",
    "uniform, stretched",
]


g = sns.FacetGrid(
    data=distributions,
    col="dataset",
    col_order=col_order,
    col_wrap=2,
    sharey=False,
    sharex=False,
    aspect=1,
    height=1.5,
)

g.map(sns.scatterplot, "x", "y", marker="+")

for ax in g.axes.flat:
    ax.set_aspect("equal", adjustable="box")
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_visible(False)

g.set_titles(template="{col_name}")
g.set_axis_labels(x_var="", y_var="")
# g.add_legend()
g.tight_layout()
g.savefig(OUT_PATH + "datasets.pdf")
