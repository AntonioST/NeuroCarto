import numpy as np

from chmap.probe_npx import NpxProbeDesp
from chmap.util.util_blueprint import BlueprintFunctions
from chmap.util.utils import TimeMarker


def blueprint_simple_init_script_from_activity_data_with_a_threshold(bp: BlueprintFunctions, arg: str):
    """\
    input: "FILE, THRESHOLD"
    FILE: (str) a numpy filepath, which shape Array[int, N, (shank, col, row, state, value)]
    THRESHOLD: (float) activities threshold to set FULL category
    """
    if len(arg) == 0:
        raise RuntimeError('empty empty. Need "FILE,THRESHOLD"')

    marker = TimeMarker(disable=False)
    bp.check_probe(NpxProbeDesp)
    marker('check_probe')

    args = [it.strip() for it in arg.split(',')]
    filename = bp.arg.get_value('filename', args, 0)
    threshold = bp.arg.get_value('threshold', args, 1, float)
    marker('arg.get_value')

    data = bp.load_data(filename)
    marker('load_data')

    F = NpxProbeDesp.CATE_FULL
    H = NpxProbeDesp.CATE_HALF
    Q = NpxProbeDesp.CATE_QUARTER
    L = NpxProbeDesp.CATE_LOW
    X = NpxProbeDesp.CATE_FORBIDDEN
    marker('categories')

    bp.log_message(f'min={np.nanmin(data)}, max={np.nanmax(data)}')
    marker('print min, max')

    data = bp.interpolate_nan(data)
    marker('interpolate_nan')
    bp.draw(data)
    marker('draw')

    bp[data >= threshold] = F
    marker('FULL')

    bp.fill(F, gap=None)
    marker('fill.0')
    bp.fill(F, threshold=10, unset=True)
    marker('fill.1')

    bp.extend(F, 2, threshold=(0, 100))
    marker('extend.0')
    bp.extend(F, 10, H)
    marker('extend.1')
