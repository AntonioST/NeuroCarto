import time
from pathlib import Path

from matplotlib import pyplot as plt

from neurocarto.probe_npx import plot
from neurocarto.probe_npx.desp import NpxProbeDesp
from neurocarto.probe_npx.npx import ChannelMap
from neurocarto.probe_npx.stat import npx_electrode_probability
from neurocarto.util.debug import print_save

file = Path('res/Fig3_example.imro')
desp = NpxProbeDesp()
chmap = ChannelMap.from_imro(file)
blueprint = desp.load_blueprint(file.with_suffix('.blueprint.npy'), desp.all_electrodes(chmap))

selector = [
    'default', 'weaker',
    'neurocarto.probe_npx.select_weaker_debug:electrode_select'
][0]
print(f'use selector {selector}')

t = time.time()
prob = npx_electrode_probability(desp, chmap, blueprint, selector=selector, sample_times=1000, n_worker=6)
t = time.time() - t
print(f'use {t:.2f} sec')
print(f'complete rate : {100 * prob.complete_rate:.2f}%')
print(f'max(Ceff) : {100 * prob.channel_efficiency:.2f}%')
print(f'mean(Ceff) : {100 * prob.channel_efficiency_mean:.2f}%')
print(f'var(Ceff) : {100 * prob.channel_efficiency_var:.2f}%')

fg, ax = plt.subplots()
ax.hist(prob.channel_efficiency_, bins=20)
plt.savefig(print_save(f'res/Fig3e_ceff_{selector}.png'))

with plt.rc_context(fname='tests/default.matplotlibrc'):
    fg, ax = plt.subplots(gridspec_kw=dict(top=0.9))
    height = 6

    ims = plot.plot_electrode_block(ax, chmap, prob.probability, electrode_unit='raw', height=height,
                                    cmap='YlOrBr', vmin=0, vmax=1, sparse=False, shank_width_scale=2)
    plot.plot_probe_shape(ax, chmap, height=height, color='gray', label_axis=True, shank_width_scale=2)

    cax = ax.inset_axes([0, 1.1, 1, 0.02])  # [x0, y0, width, height]
    ax.figure.colorbar(ims[0], cax=cax, orientation='horizontal')

    plt.savefig(print_save('res/Fig3e_prob.png'))

"""
use selector default
use 35.48 sec
complete rate : 98.80%
max(Ceff) : 87.95%
mean(Ceff) : 85.76%
var(Ceff) : 0.01%

use selector weaker
use 21.28 sec
complete rate : 100.00%
max(Ceff) : 92.33%
mean(Ceff) : 79.60%
var(Ceff) : 0.07%
"""
