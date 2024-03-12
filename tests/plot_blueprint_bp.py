"""
Test file for testing algorithm in neurocarto.views.blueprint:BlueprintView.
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

from neurocarto.probe_npx.desp import NpxProbeDesp
from neurocarto.probe_npx.npx import ChannelMap
from neurocarto.util.util_blueprint import BlueprintFunctions
from neurocarto.util.utils import TimeMarker

marker = TimeMarker()
marker('import')

rc = matplotlib.rc_params_from_file('tests/default.matplotlibrc', fail_on_error=True, use_default_template=True)
plt.rcParams.update(rc)
marker('load rc')

file = Path(sys.argv[1])
desp = NpxProbeDesp()
chmap = ChannelMap.from_imro(file)
blueprint = desp.load_blueprint(file.with_suffix('.blueprint.npy'), chmap)
marker('load files')

bp = BlueprintFunctions.from_blueprint(blueprint, desp.all_possible_categories())
blueprint = bp.set(bp.blueprint(), NpxProbeDesp.CATE_SET, NpxProbeDesp.CATE_FULL)
categories = [
    NpxProbeDesp.CATE_FULL, NpxProbeDesp.CATE_HALF, NpxProbeDesp.CATE_QUARTER, NpxProbeDesp.CATE_FORBIDDEN
]
marker('setup bp')

edges = bp.clustering_edges(blueprint, categories)
marker('bp.clustering_edges')

probe_type = chmap.probe_type
w = probe_type.c_space // 2
h = probe_type.r_space // 2
edges = [it.set_corner((w, h)) for it in edges]
marker('bp.set_corner')

xs = [[], [], [], []]
ys = [[], [], [], []]
color = ['green', 'orange', 'blue', 'pink']

for edge in edges:
    i = categories.index(edge.category)
    xs[i].append([edge.x])
    ys[i].append([edge.y])

marker('ret')

fg, ax = plt.subplots()
for i in range(len(color)):
    cc = color[i]
    for xx, yy in zip(xs[i], ys[i]):
        # ignore holes
        xx = xx[0]
        yy = yy[0]
        art = Polygon(np.vstack([xx, yy]).T, closed=True, color=cc)
        ax.add_artist(art)

ax.set_xlim(-1, 1000)
ax.set_ylim(-1, 10000)

marker('plot')
plt.show()
