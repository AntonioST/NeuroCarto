import numpy as np

from chmap.probe_npx import NpxProbeDesp, utils
from chmap.util.util_blueprint import BlueprintFunctions
from chmap.util.utils import TimeMarker


def npx24_single_shank(bp: BlueprintFunctions, shank: int = 0, row: int = 0):
    """
    Make a block channelmap for 4-shank Neuropixels probe.

    :param bp:
    :param shank: (int=0) on which shank.
    :param row: (int=0) start row in um.
    """
    bp.check_probe(NpxProbeDesp, 24)
    bp.set_channelmap(utils.npx24_single_shank(shank, row, um=True))


def npx24_stripe(bp: BlueprintFunctions, row: int = 0):
    """
    Make a block channelmap for 4-shank Neuropixels probe.

    :param bp:
    :param row: (int=0) start row in um.
    """
    bp.check_probe(NpxProbeDesp, 24)
    bp.set_channelmap(utils.npx24_stripe(row, um=True))


def npx24_half_density(bp: BlueprintFunctions, shank: int | list[int] = 0, row: int = 0):
    """
    Make a channelmap for 4-shank Neuropixels probe that uniformly distributes channels in *half* density.

    :param bp:
    :param shank: (int|[int, int]=0) on which shank/s.
    :param row: (int=0) start row in um.
    """
    bp.check_probe(NpxProbeDesp, 24)
    bp.set_channelmap(utils.npx24_half_density(shank, row, um=True))


def npx24_quarter_density(bp: BlueprintFunctions, shank: int | list[int] | None = None, row: int = 0):
    """
    Make a channelmap for 4-shank Neuropixels probe that uniformly distributes channels in *quarter* density.

    :param bp:
    :param shank: (int|[int, int]=None) on which shank/s. use `None` for four shanks.
    :param row: (int=0) start row in um.
    """
    bp.check_probe(NpxProbeDesp, 24)
    bp.set_channelmap(utils.npx24_quarter_density(shank, row, um=True))


def npx24_one_eighth_density(bp: BlueprintFunctions, row: int = 0):
    """
    Make a channelmap for 4-shank Neuropixels probe that uniformly distributes channels in *one-eighth* density.

    :param bp:
    :param row: (int=0) start row in um.
    """
    bp.check_probe(NpxProbeDesp, 24)
    bp.set_channelmap(utils.npx24_one_eighth_density(row, um=True))


def move_blueprint(bp: BlueprintFunctions, y: int, shank: list[int] = None, update=False):
    """
    Move blueprint upward or downward.

    :param bp:
    :param y: (int) um
    :param shank: (list[int]=None) only particular shanks.
    :param update: (bool) update channelmap to follow the blueprint change.
    """
    bp.check_probe()

    if shank is None:
        mask = None
    else:
        mask = np.zeros_like(bp.s, dtype=bool)
        for s in shank:
            np.logical_or(mask, bp.s == s, out=mask)

    bp.set_blueprint(bp.move(bp.blueprint(), ty=y, mask=mask, axis=0, init=bp.CATE_UNSET))
    if update:
        bp.refresh_selection()


def exchange_shank(bp: BlueprintFunctions, shank: list[int], update=False):
    """
    Move blueprint between shanks.

    *Note*: Each shank requires the same electrode number, and electrodes are ordered consistently.

    :param bp:
    :param shank: (list[int]): For N shank probe, it is an N-length list.
        For example, `[3, 2, 1, 0]` gives a reverse-shank-ordered blueprint.
    :param update: (bool) update channelmap to follow the blueprint change.
    """
    bp.check_probe()

    ns = np.max(bp.s) + 1
    if len(shank) != np.max(ns):
        raise RuntimeError(f'not a {ns}-length list: {shank}')

    p = bp.blueprint()
    q = bp.new_blueprint()
    for i, s in enumerate(shank):
        q[bp.s == i] = p[bp.s == s]

    bp.set_blueprint(q)
    if update:
        bp.refresh_selection()


def load_blueprint(bp: BlueprintFunctions, filename: str):
    """
    Load a blueprint file.

    :param bp:
    :param filename: (str) a numpy file '*.blueprint.npy'.
    """
    bp.check_probe()

    bp.set_blueprint(bp.probe.load_blueprint(filename, bp.channelmap))


def enable_electrode_as_pre_selected(bp: BlueprintFunctions):
    """
    Set captured electrodes as *pre-selected* category.

    :param bp:
    """
    bp.check_probe()
    bp.set_blueprint(bp.set(bp.blueprint(), bp.captured_electrodes(), bp.CATE_SET))


def blueprint_simple_init_script_from_activity_data_with_a_threshold(bp: BlueprintFunctions, filename: str, threshold: float):
    """
    Initial a blueprint based on the experimental activity data with a given threshold,
    which follows:

    * set NaN area as forbidden area.
    * set full-density to the area which corresponding activity over the threshold.
    * make the full-density area into rectangle by filling the gaps.
    * extend the full-density area with half-density.

    :param bp:
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

    bp[np.isnan(data)] = X
    marker('NaN')
    bp.reduce(X, 20, bi=False)
    marker('reduce')
    bp.fill(X, gap=None, threshold=10, unset=True)
    marker('reduce.fill')

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
