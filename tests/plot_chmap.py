import sys

import matplotlib
from matplotlib import pyplot as plt

from chmap.probe_npx.npx import ChannelMap
from chmap.probe_npx.plot import plot_channelmap_block, plot_probe_shape

rc = matplotlib.rc_params_from_file('tests/default.matplotlibrc', fail_on_error=True, use_default_template=True)
plt.rcParams.update(rc)

file = sys.argv[1]
chmap = ChannelMap.from_imro(file)

fg, ax = plt.subplots()
plot_channelmap_block(ax, chmap, height=3.2, color='k', shank_width_scale=2)
plot_probe_shape(ax, chmap.probe_type, height=3.2, color='gray', label_axis=True, shank_width_scale=2)

if len(sys.argv) == 2:
    plt.show()
else:
    plt.savefig(sys.argv[2])
