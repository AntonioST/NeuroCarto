from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

from neurocarto.probe_npx.npx import ChannelMap
from neurocarto.probe_npx.plot import plot_channelmap_block, plot_probe_shape
from neurocarto.util.debug import print_save

rc = matplotlib.rc_params_from_file('tests/default.matplotlibrc', fail_on_error=True, use_default_template=True)
plt.rcParams.update(rc)

files = [
    Path('res/shank0_halfdensity.imro'),
    Path('res/shank1_halfdensity.imro'),
    Path('res/shank2_halfdensity.imro'),
    Path('res/shank3_halfdensity.imro'),
]

kwargs = dict(height=6, shank_width_scale=2, reverse_shank=True)
colors = ['r', 'orange', 'g', 'b']

fg, ax = plt.subplots()
for i, file in enumerate(files):
    plot_channelmap_block(ax, chmap := ChannelMap.from_imro(file), color=colors[i], **kwargs)
plot_probe_shape(ax, chmap, label_axis=True, **kwargs)

plt.savefig(print_save('res/Fig5c_channelmap.png'))
