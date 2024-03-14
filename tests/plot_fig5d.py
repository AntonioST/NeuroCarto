from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

from neurocarto.probe_npx.npx import ChannelMap
from neurocarto.probe_npx.plot import plot_channelmap_block, plot_probe_shape
from neurocarto.util.utils import print_save

rc = matplotlib.rc_params_from_file('tests/default.matplotlibrc', fail_on_error=True, use_default_template=True)
plt.rcParams.update(rc)

file = Path('res/animal_TS42_A02_20230820.imro')
chmap = ChannelMap.from_imro(file)

kwargs = dict(height=6, shank_width_scale=2, reverse_shank=True)

fg, ax = plt.subplots()
plot_channelmap_block(ax, chmap, color='k', **kwargs)
plot_probe_shape(ax, chmap, label_axis=True, **kwargs)

plt.savefig(print_save('res/Fig5d_channelmap.png'))
