import numpy as np

from chmap.probe_npx import NpxProbeDesp
from chmap.util.util_blueprint import BlueprintFunctions


def blueprint_simple_init_script_from_activity_data_with_a_threshold(bp: BlueprintFunctions, arg: str):
    """\
    input: "FILE, THRESHOLD"
    FILE: (str) a numpy filepath, which shape Array[int, N, (shank, col, row, state, value)]
    THRESHOLD: (float) activities threshold to set FULL category
    """
    if len(arg) == 0:
        raise RuntimeError('empty empty. Need "FILE,THRESHOLD"')

    bp.check_probe(NpxProbeDesp)

    args = [it.strip() for it in arg.split(',')]
    filename = bp.arg.get_value('filename', args, 0)
    threshold = bp.arg.get_value('threshold', args, 1, float)

    data = bp.load_data(filename)

    F = NpxProbeDesp.CATE_FULL
    H = NpxProbeDesp.CATE_HALF
    Q = NpxProbeDesp.CATE_QUARTER
    L = NpxProbeDesp.CATE_LOW
    X = NpxProbeDesp.CATE_FORBIDDEN

    bp.log_message(f'min={np.nanmin(data)}, max={np.nanmax(data)}')

    data = bp.interpolate_nan(data)
    bp.draw(data)

    bp[data >= threshold] = F

    bp.fill(F, gap=None)
    bp.fill(F, threshold=10, unset=True)
    bp.extend(F, 2, threshold=(0, 100))
    bp.extend(F, 10, H)
