import sys
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

from neurocarto.probe_npx.desp import NpxProbeDesp
from neurocarto.probe_npx.npx import ChannelMap
from neurocarto.probe_npx.plot import plot_category_area, plot_probe_shape

rc = matplotlib.rc_params_from_file('tests/default.matplotlibrc', fail_on_error=True, use_default_template=True)
plt.rcParams.update(rc)

file = Path(sys.argv[1])
desp = NpxProbeDesp()
chmap = ChannelMap.from_imro(file)
blueprint = desp.load_blueprint(file.with_suffix('.blueprint.npy'), chmap)

fg, ax = plt.subplots()
height = 6
plot_category_area(ax, chmap, blueprint, height=height, shank_width_scale=2)
plot_probe_shape(ax, chmap, height=height, color='gray', label_axis=True, shank_width_scale=2)

if len(sys.argv) == 2:
    plt.show()
else:
    plt.savefig(sys.argv[2], dpi=600)
