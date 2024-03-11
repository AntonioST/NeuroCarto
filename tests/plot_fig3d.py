from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

from chmap.probe_npx.desp import NpxProbeDesp
from chmap.probe_npx.npx import ChannelMap
from chmap.util.util_blueprint import BlueprintFunctions

rc = matplotlib.rc_params_from_file('tests/default.matplotlibrc', fail_on_error=True, use_default_template=True)
plt.rcParams.update(rc)

file = Path('res/Fig3_example.imro')
D = NpxProbeDesp()
chmap = ChannelMap.from_imro(file)
bp = BlueprintFunctions(D, chmap)
bp.load_blueprint(file.with_suffix('.blueprint.npy'))

kwargs = dict(height=6, probe_color='k', shank_width_scale=2, label_axis=True)

# ----------------------------------------------------------------------------------------------------------------------

fg, ax = plt.subplots()
bp.plot_channelmap(ax=ax, **kwargs)
plt.savefig('res/Fig3_example.png')

# ----------------------------------------------------------------------------------------------------------------------

fg, ax = plt.subplots()
bp.plot_blueprint(ax=ax, **kwargs)
plt.savefig('res/Fig3_example.policy.png')  # old name

# ----------------------------------------------------------------------------------------------------------------------

blueprint = bp.blueprint()
blueprint = bp.unset(blueprint, [D.CATE_SET, D.CATE_HALF, D.CATE_QUARTER, D.CATE_LOW, D.CATE_FORBIDDEN])
blueprint = bp.invalid(blueprint, categories=D.CATE_FULL, value=D.CATE_FORBIDDEN)

fg, ax = plt.subplots()
bp.plot_blueprint(blueprint, {D.CATE_FULL: 'green', D.CATE_FORBIDDEN: 'red'}, ax=ax, **kwargs)
ax.set_title(f'{bp.count_categories(blueprint, D.CATE_FULL)}/384')
plt.savefig('res/Fig3d-conflict.png')

# ----------------------------------------------------------------------------------------------------------------------

blueprint = bp.blueprint()
full_mask = blueprint == D.CATE_FULL
half_mask = blueprint == D.CATE_HALF
quar_mask = blueprint == D.CATE_QUARTER
selected = bp.selected_electrodes()
blueprint = bp.new_blueprint()
blueprint[selected] = D.CATE_SET
blueprint[~(full_mask | half_mask)] = D.CATE_UNSET
blueprint = bp.invalid(blueprint, categories=D.CATE_SET, value=D.CATE_FORBIDDEN)

fg, ax = plt.subplots()
bp.plot_blueprint(blueprint, {D.CATE_SET: 'green', D.CATE_FORBIDDEN: 'red'}, ax=ax, **kwargs)
ax.set_title(f'{bp.count_categories(blueprint, D.CATE_SET)}/384')
plt.savefig('res/Fig3d-conflict-half.png')

# ----------------------------------------------------------------------------------------------------------------------

blueprint = bp.new_blueprint()
blueprint[selected] = D.CATE_SET
blueprint[~(full_mask | half_mask | quar_mask)] = D.CATE_UNSET
blueprint = bp.invalid(blueprint, categories=D.CATE_SET, value=D.CATE_FORBIDDEN)

fg, ax = plt.subplots()
bp.plot_blueprint(blueprint, {D.CATE_SET: 'green', D.CATE_FORBIDDEN: 'red'}, ax=ax, **kwargs)
ax.set_title(f'{bp.count_categories(blueprint, D.CATE_SET)}/384')
plt.savefig('res/Fig3d-conflict-quarter.png')

# ----------------------------------------------------------------------------------------------------------------------

# plt.show()
