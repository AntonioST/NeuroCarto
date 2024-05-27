from __future__ import annotations

import sys
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from neurocarto.probe_npx import NpxProbeDesp, NpxElectrodeDesp
from neurocarto.probe_npx.npx import ChannelMap
from neurocarto.probe_npx.select import ElectrodeSelector, load_select
from neurocarto.util.util_blueprint import BlueprintFunctions
from neurocarto.util.utils import doc_link

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = [
    'npx_electrode_density',
    'npx_request_electrode',
    'npx_channel_efficiency',
    'ElectrodeProbability',
    'npx_electrode_probability'
]


def npx_electrode_density(chmap: ChannelMap) -> NDArray[np.float_]:
    """

    :param chmap:
    :return: density curve array. Array[float, S, (v, y), Y].
    """
    from neurocarto.probe_npx.plot import cast_electrode_curve

    probe_type = chmap.probe_type
    bp = BlueprintFunctions(NpxProbeDesp(), chmap)
    data = np.zeros(len(bp))
    data[bp.selected_electrodes(chmap)] += probe_type.r_space / probe_type.n_col_shank
    data, y = cast_electrode_curve(probe_type, data, electrode_unit='raw', kernel='norm')
    ns, nr = data.shape
    ir = np.arange(0, nr, 7)
    ret = np.zeros((ns, 2, len(ir)))
    ret[:, 0, :] = data[:, ir]
    ret[:, 1, :] = y[ir]
    return ret


@doc_link()
def npx_request_electrode(bp: BlueprintFunctions, blueprint: NDArray[np.int_] = None) -> float:
    """

    :param bp:
    :param blueprint: a given blueprint.
    :return: channel efficiency value
    """
    if blueprint is None:
        blueprint = bp._blueprint

    electrode = 0
    for category, count in zip(*np.unique(blueprint, return_counts=True)):
        match category:
            case NpxProbeDesp.CATE_SET | NpxProbeDesp.CATE_FULL:
                electrode += count
            case NpxProbeDesp.CATE_HALF:
                electrode += count / 2
            case NpxProbeDesp.CATE_QUARTER:
                electrode += count / 4

    return electrode


@doc_link()
def npx_channel_efficiency(bp: BlueprintFunctions,
                           channelmap: ChannelMap = None,
                           blueprint: NDArray[np.int_] = None) -> tuple[float, float]:
    """
    Calculate the area and channel efficiency for a channel map with a given blueprint.

    :param bp:
    :param channelmap: channelmap outcomes from *blueprint*
    :param blueprint: a given blueprint.
    :return: tuple of area and channel efficiency value
    """
    if channelmap is None:
        channelmap = bp.channelmap

    if blueprint is None:
        blueprint = bp._blueprint

    electrode = npx_request_electrode(bp, blueprint)
    total = channelmap.probe_type.n_channels
    unused = total - len(channelmap)
    channel = 0
    excluded = 0

    selected = blueprint[bp.selected_electrodes(channelmap)]
    for category, count in zip(*np.unique(selected, return_counts=True)):
        match category:
            case NpxProbeDesp.CATE_SET | NpxProbeDesp.CATE_FULL | NpxProbeDesp.CATE_HALF | NpxProbeDesp.CATE_QUARTER:
                channel += count
            case NpxProbeDesp.CATE_EXCLUDED:
                excluded += count

    ae = 0 if electrode == 0 else max(channel / electrode, 0)
    ce = 0 if ae == 0 else min(ae, 1 / ae)
    ex = (total - excluded - unused) / total
    return ae, ce * ex


class ElectrodeProbability(NamedTuple):
    sample_times: int
    """number of sample times"""
    summation: NDArray[np.int_]
    """summation matrix Array[count:int, S, C, R]"""
    complete: int
    """number of sample that get a complete result"""
    channel_efficiency_: NDArray[np.float_]
    """collected channel_efficiency array."""

    @property
    def probability(self) -> NDArray[np.float_]:
        """probability matrix Array[prob:float, S, C, R]"""
        return self.summation.astype(float) / self.sample_times

    @property
    def complete_rate(self) -> float:
        return self.complete / self.sample_times

    @property
    def channel_efficiency(self) -> float:
        """max channel efficiency"""
        return np.max(self.channel_efficiency_)

    @property
    def channel_efficiency_mean(self) -> float:
        """mean channel efficiency"""
        return np.mean(self.channel_efficiency_)

    @property
    def channel_efficiency_var(self) -> float:
        """channel efficiency variance"""
        return np.var(self.channel_efficiency_)

    def __add__(self, other: ElectrodeProbability) -> Self:
        return ElectrodeProbability(
            self.sample_times + other.sample_times,
            self.summation + other.summation,
            self.complete + other.complete,
            np.concatenate([self.channel_efficiency_, other.channel_efficiency_]),
        )

    @classmethod
    def _reduce_add(cls, result: list[ElectrodeProbability]) -> Self:
        return ElectrodeProbability(
            sum([it.sample_times for it in result]),
            np.sum([it.summation for it in result], axis=0),
            sum([it.complete for it in result]),
            np.concatenate([it.channel_efficiency_ for it in result]),
        )


def npx_electrode_probability(probe: NpxProbeDesp, chmap: ChannelMap, blueprint: list[NpxElectrodeDesp],
                              selector: str | ElectrodeSelector = 'default',
                              sample_times: int = 1000,
                              n_worker: int = 1) -> ElectrodeProbability:
    """
    Sample *sample_times* channelmap outcomes for a given *blueprint*.

    :param probe:
    :param chmap: channelmap instance, use as a reference.
    :param blueprint: a given blueprint.
    :param selector: use which electrode selecting method.
    :param sample_times:
    :param n_worker: number of process.
    :return: ElectrodeProbability
    """
    if isinstance(selector, str):
        selector = load_select(selector)

    if n_worker == 1:
        return _npx_electrode_probability_0(probe, chmap, blueprint, selector, sample_times)
    else:
        return _npx_electrode_probability_n(probe, chmap, blueprint, selector, sample_times, n_worker=n_worker)


def _npx_electrode_probability_0(probe: NpxProbeDesp, chmap: ChannelMap, blueprint: list[NpxElectrodeDesp],
                                 selector: ElectrodeSelector,
                                 sample_times: int) -> ElectrodeProbability:
    pt = chmap.probe_type
    bp = BlueprintFunctions(probe, chmap)
    mat = np.zeros((pt.n_shank, pt.n_col_shank, pt.n_row_shank))
    complete = 0
    channel_efficiency = []

    for _ in range(sample_times):
        chmap = selector(probe, chmap, blueprint)

        for t in chmap.electrodes:
            mat[t.shank, t.column, t.row] += 1

        if probe.is_valid(chmap):
            complete += 1

        bp.set_blueprint(blueprint)
        channel_efficiency.append(npx_channel_efficiency(bp, chmap)[1])

    return ElectrodeProbability(sample_times, mat, complete, np.array(channel_efficiency))


def _npx_electrode_probability_n(probe: NpxProbeDesp, chmap: ChannelMap, blueprint: list[NpxElectrodeDesp],
                                 selector: ElectrodeSelector,
                                 sample_times: int,
                                 n_worker: int) -> ElectrodeProbability:
    if n_worker <= 1:
        raise ValueError()

    sample_times_per_worker = sample_times // n_worker
    sample_times_list = [sample_times_per_worker] * n_worker
    sample_times_list[-1] += sample_times - sum(sample_times_list)
    assert sum(sample_times_list) == sample_times

    import multiprocessing
    with multiprocessing.Pool(n_worker) as pool:
        jobs = []
        for _sample_times in sample_times_list:
            jobs.append(pool.apply_async(_npx_electrode_probability_0, (probe, chmap, blueprint, selector, _sample_times)))
        pool.close()
        pool.join()

    return ElectrodeProbability._reduce_add([it.get() for it in jobs])
