import sys
import time
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

from chmap.probe_npx import plot
from chmap.probe_npx.desp import NpxProbeDesp
from chmap.probe_npx.npx import ChannelMap
from chmap.probe_npx.stat import npx_electrode_probability

rc = matplotlib.rc_params_from_file('tests/default.matplotlibrc', fail_on_error=True, use_default_template=True)
plt.rcParams.update(rc)

file = Path(sys.argv[1])
desp = NpxProbeDesp()
chmap = ChannelMap.from_imro(file)
blueprint = desp.load_blueprint(file.with_suffix('.blueprint.npy'), desp.all_electrodes(chmap))

selector = ['default', 'weaker'][0]
print(f'use selector {selector}')

t = time.time()
prob = npx_electrode_probability(desp, chmap, blueprint, selector=selector, sample_times=100, n_worker=6)
t = time.time() - t
print(f'use {t:.2f} sec')
print(f'complete rate : {100 * prob.complete_rate:.2f}%')
print(f'max(Ceff) : {100 * prob.channel_efficiency:.2f}%')
print(f'mean(Ceff) : {100 * prob.channel_efficiency_mean:.2f}%')
print(f'var(Ceff) : {100 * prob.channel_efficiency_var:.2f}%')

fg, ax = plt.subplots(gridspec_kw=dict(top=0.9))
height = 6

plot.plot_electrode_matrix(ax, chmap.probe_type, prob, electrode_unit='raw', cmap='YlOrBr', vmin=0, vmax=1)
ims = plot.plot_electrode_block(ax, chmap.probe_type, prob.probability, electrode_unit='raw', height=height,
                                cmap='YlOrBr', vmin=0, vmax=1, shank_width_scale=2)
plot.plot_probe_shape(ax, chmap.probe_type, height=height, color='gray', label_axis=True, shank_width_scale=2)

cax = ax.inset_axes([0, 1.1, 1, 0.02])  # [x0, y0, width, height]
ax.figure.colorbar(ims[0], cax=cax, orientation='horizontal', )

if len(sys.argv) == 2:
    plt.show()
else:
    plt.savefig(sys.argv[2])
