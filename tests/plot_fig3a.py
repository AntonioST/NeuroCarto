import matplotlib
from matplotlib import pyplot as plt

from neurocarto.probe_npx.plot import plot_channelmap_block, plot_probe_shape
from neurocarto.probe_npx.utils import *
from neurocarto.util.debug import print_save

rc = matplotlib.rc_params_from_file('tests/default.matplotlibrc', fail_on_error=True, use_default_template=True)
plt.rcParams.update(rc)

h1 = npx24_half_density(0)
h2 = npx24_half_density((0, 1))
q2 = npx24_quarter_density((0, 1))
q4 = npx24_quarter_density(None)
o1 = npx24_one_eighth_density()

chmaps = [h1, h2, q2, q4, o1]

figsize_w, figsize_h = rc['figure.figsize']

fg, ax = plt.subplots(
    ncols=len(chmaps),
    figsize=(2 * len(chmaps), figsize_h),
    sharey='all',
    sharex='all',
    squeeze=True
)

height = 6
for i, chmap in enumerate(chmaps):
    plot_channelmap_block(ax[i], chmap, height=height, color='k', shank_width_scale=2)
    plot_probe_shape(ax[i], chmap.probe_type, height=height, color='gray', label_axis=True, shank_width_scale=2)

    if i != 0:
        ax[i].set_ylabel(None)
    ax[i].set_xlabel(None)
    ax[i].set_xticklabels([])

plt.savefig(print_save('res/Fig3_uniform.png'))
