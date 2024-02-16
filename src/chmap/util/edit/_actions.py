import numpy as np

from chmap.probe_npx import NpxProbeDesp, utils
from chmap.util.util_blueprint import BlueprintFunctions
from chmap.util.utils import TimeMarker


def npx24_single_shank(bp: BlueprintFunctions, shank: int = 0, row: int = 0):
    """\
    Make a block channelmap for 4-shank Neuropixels probe.

    :param shank: (int=0) on which shank.
    :param row: (int=0) start row in um.
    """
    bp.check_probe(NpxProbeDesp, 24)
    bp.set_channelmap(utils.npx24_single_shank(shank, row, um=True))


def npx24_stripe(bp: BlueprintFunctions, row: int = 0):
    """\
    Make a block channelmap for 4-shank Neuropixels probe.

    :param row: (int=0) start row in um.
    """
    bp.check_probe(NpxProbeDesp, 24)
    bp.set_channelmap(utils.npx24_stripe(row, um=True))


def npx24_half_density(bp: BlueprintFunctions, shank: int | list[int] = 0, row: int = 0):
    """\
    Make a channelmap for 4-shank Neuropixels probe that uniformly distributes channels in half density.

    :param shank: (int|[int, int]=0) on which shank/s.
    :param row: (int=0) start row in um.
    """
    bp.check_probe(NpxProbeDesp, 24)
    bp.set_channelmap(utils.npx24_half_density(shank, row, um=True))


def npx24_quarter_density(bp: BlueprintFunctions, shank: int | list[int] | None = None, row: int = 0):
    """\
    Make a channelmap for 4-shank Neuropixels probe that uniformly distributes channels in quarter density.

    :param shank: (int|[int, int]=None) on which shank/s. use `None` for four shanks.
    :param row: (int=0) start row in um.
    """
    bp.check_probe(NpxProbeDesp, 24)
    bp.set_channelmap(utils.npx24_quarter_density(shank, row, um=True))


def npx24_one_eighth_density(bp: BlueprintFunctions, row: int = 0):
    """\
    Make a channelmap for 4-shank Neuropixels probe that uniformly distributes channels in one-eighth density.

    :param row: (int=0) start row in um.
    """
    bp.check_probe(NpxProbeDesp, 24)
    bp.set_channelmap(utils.npx24_one_eighth_density(row, um=True))


def blueprint_simple_init_script_from_activity_data_with_a_threshold(bp: BlueprintFunctions, filename: str, threshold: float):
    """\
    :param filename: a numpy filepath, which shape Array[int, N, (shank, col, row, state, value)]
    :param threshold: (float) activities threshold to set FULL category
    """
    marker = TimeMarker(disable=False)
    bp.check_probe(NpxProbeDesp)
    marker('check_probe')

    bp.log_message(f'{filename=}', f'{threshold=}')
    marker('log args')

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
