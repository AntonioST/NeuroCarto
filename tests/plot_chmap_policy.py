import sys
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from chmap.probe_npx.desp import NpxProbeDesp
from chmap.probe_npx.npx import ChannelMap
from chmap.probe_npx.plot import plot_electrode_block, plot_probe_shape

rc = matplotlib.rc_params_from_file('tests/default.matplotlibrc', fail_on_error=True, use_default_template=True)
plt.rcParams.update(rc)

file = Path(sys.argv[1])
desp = NpxProbeDesp()
chmap = ChannelMap.from_imro(file)
policy = np.load(file.with_suffix('.policy.npy'))
electrodes = desp.electrode_from_numpy(desp.all_electrodes(chmap), policy)


def get_electrode_under_policy(*policies: int) -> NDArray[np.int_]:
    ret = []
    for e in electrodes:
        if e.policy in policies:
            ret.append(e.electrode)

    return np.array(ret).reshape(-1, 3)


fg, ax = plt.subplots()
height = 6

if len(e := get_electrode_under_policy(NpxProbeDesp.POLICY_SET, NpxProbeDesp.POLICY_D1)) > 0:
    plot_electrode_block(ax, chmap.probe_type, e, height=height, color='green', shank_width_scale=2)
if len(e := get_electrode_under_policy(NpxProbeDesp.POLICY_D2)) > 0:
    plot_electrode_block(ax, chmap.probe_type, e, height=height, color='orange', shank_width_scale=2)
if len(e := get_electrode_under_policy(NpxProbeDesp.POLICY_D4)) > 0:
    plot_electrode_block(ax, chmap.probe_type, e, height=height, color='blue', shank_width_scale=2)
if len(e := get_electrode_under_policy(NpxProbeDesp.POLICY_REMAINDER)) > 0:
    plot_electrode_block(ax, chmap.probe_type, e, height=height, color='gray', shank_width_scale=2)
if len(e := get_electrode_under_policy(NpxProbeDesp.POLICY_FORBIDDEN)) > 0:
    plot_electrode_block(ax, chmap.probe_type, e, height=height, color='pink', shank_width_scale=2)

plot_probe_shape(ax, chmap.probe_type, height=height, color='gray', label_axis=True, shank_width_scale=2)

if len(sys.argv) == 2:
    plt.show()
else:
    plt.savefig(sys.argv[2])
