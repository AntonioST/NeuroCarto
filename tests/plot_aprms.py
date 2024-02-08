import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from chmap.probe_npx import plot
from chmap.probe_npx.npx import PROBE_TYPE_NP24

rc = matplotlib.rc_params_from_file('tests/default.matplotlibrc', fail_on_error=True, use_default_template=True)
plt.rcParams.update(rc)

data = np.load('res/Fig5d_data.npy')  # Array[int, N, (S, C, R, _, V)]
data = np.delete(data, 3, 1)

fg, ax = plt.subplots()
# plot.plot_electrode_matrix(
#     ax, PROBE_TYPE_NP24, data,
#     electrode_unit='cr',
#     kernel=(0, 1),
#     shank_list=[3, 2, 1, 0],
#     cmap='plasma'
# )

plot.plot_electrode_block(
    ax, PROBE_TYPE_NP24,
    plot.ElectrodeMatData.of(PROBE_TYPE_NP24, data, electrode_unit='cr').interpolate_nan((0, 1)),
    cmap='plasma'
)
plot.plot_probe_shape(ax, PROBE_TYPE_NP24, color='k')

plt.show()
