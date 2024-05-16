from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from neurocarto.probe_npx.desp import NpxProbeDesp
from neurocarto.probe_npx.npx import ChannelMap
from neurocarto.probe_npx.stat import npx_channel_efficiency
from neurocarto.util.debug import TimeMarker
from neurocarto.util.edit.optimize import generate_channelmap
from neurocarto.util.util_blueprint import BlueprintFunctions

file = Path('res/Fig3_example.imro')
desp = NpxProbeDesp()
chmap = ChannelMap.from_imro(file)
bp = BlueprintFunctions(desp, chmap)
blueprint = bp.load_blueprint(file.with_suffix('.blueprint.npy'))

aef, cef = npx_channel_efficiency(bp, chmap, blueprint)
print(f'Aeff={aef}')
print(f'Ceff={cef}')

sample_times = 400

t = TimeMarker()
t.reset('generate_channelmap')
maps, aefs, cefs = generate_channelmap(bp, chmap, blueprint, sample_times=sample_times, n_worker=4)
t(f'generate_channelmap({sample_times=})')

# ----------------------------------------------------------------------------------------------------------------------

kwargs = dict(height=6, probe_color='k', shank_width_scale=2, label_axis=True)

# ----------------------------------------------------------------------------------------------------------------------

fg, ax = plt.subplots()
ax.hist(cefs, bins=np.linspace(0.5, 1.0, 100))
plt.show()

# ----------------------------------------------------------------------------------------------------------------------

fg, ax = plt.subplots()
ax.hist(aefs, bins=np.linspace(0.5, 1.0, 100))
plt.show()

# ----------------------------------------------------------------------------------------------------------------------

i = int(np.argmax(cefs))
max_map = maps[i]
aef = aefs[i]
cef = cefs[i]
print(f'Aeff={aef}')
print(f'Ceff={cef}')

with plt.rc_context(fname='tests/default.matplotlibrc'):
    fg, ax = plt.subplots()
    bp.plot_channelmap(ax=ax, **kwargs)
    plt.show()

# ----------------------------------------------------------------------------------------------------------------------
