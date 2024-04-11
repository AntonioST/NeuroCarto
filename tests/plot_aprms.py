import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from neurocarto.probe_npx import plot
from neurocarto.probe_npx.npx import PROBE_TYPE_NP24

rc = matplotlib.rc_params_from_file('tests/default.matplotlibrc', fail_on_error=True, use_default_template=True)
plt.rcParams.update(rc)

data_file = 'res/Fig5d_data.npy'
data = np.load(data_file)  # Array[int, N, (S, C, R, _, V)]
data = np.delete(data, 3, 1)

fg, ax = plt.subplots()

## plot_electrode_matrix

plot.plot_electrode_matrix(
    ax, PROBE_TYPE_NP24, data,
    electrode_unit='cr',
    kernel=(0, 1),
    shank_list=[3, 2, 1, 0],
    cmap='plasma'
)

## BlueprintFunctions.plot_electrode_matrix

# from neurocarto.util.util_blueprint import BlueprintFunctions
# from neurocarto.probe_npx import NpxProbeDesp, ChannelMap
#
# bp = BlueprintFunctions(NpxProbeDesp(), ChannelMap(24))
# data = bp.load_data(data_file)
# data = bp.interpolate_nan(data, kernel=1)
# data = bp.interpolate_nan(data, kernel=1)
#
# plot.plot_electrode_matrix(
#     ax, PROBE_TYPE_NP24, data,
#     electrode_unit='raw',
#     # kernel=(0, 1),
#     shank_list=[3, 2, 1, 0],
#     cmap='plasma'
# )

## plot_electrode_block

# plot.plot_electrode_block(
#     ax, PROBE_TYPE_NP24,
#     plot.ElectrodeMatData.of(PROBE_TYPE_NP24, data, electrode_unit='cr').interpolate_nan((0, 1)),
#     cmap='plasma'
# )
# plot.plot_probe_shape(ax, PROBE_TYPE_NP24, color='k')

##
plt.show()
