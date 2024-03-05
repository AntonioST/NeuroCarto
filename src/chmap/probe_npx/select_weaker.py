"""
Neuropixels another electrode selection method.
It has a *weaker* local density rule compared to the default one.
"""
from __future__ import annotations

from collections.abc import Iterator
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from .desp import NpxProbeDesp, NpxElectrodeDesp, K
from .npx import ChannelMap, ProbeType

__all__ = ['electrode_select']


def electrode_select(desp: NpxProbeDesp, chmap: ChannelMap, blueprint: list[NpxElectrodeDesp],
                     **kwargs) -> ChannelMap:
    probe_type = chmap.probe_type

    s = Struct.new(desp, chmap)
    s.init_blueprint(blueprint)
    s.init_probability()

    for e in np.random.permutation(np.nonzero(s.categories == NpxProbeDesp.CATE_SET)[0]):
        s.add(e)

    _select_loop(probe_type, s)

    return build_channelmap(desp, chmap, s)


class Struct(NamedTuple):
    electrodes: list[NpxElectrodeDesp]
    index: dict[K, int]
    categories: NDArray[np.int_]
    channels: NDArray[np.int_]
    probability: NDArray[np.float_]

    @classmethod
    def new(cls, desp: NpxProbeDesp, chmap: ChannelMap):
        electrodes = desp.all_electrodes(chmap)
        categories = np.full((len(electrodes),), desp.CATE_UNSET, dtype=int)
        probability = np.full((len(electrodes),), 0.0, dtype=float)
        channels = np.array([it.channel for it in electrodes])
        index: dict[K, int] = {
            it.electrode: i
            for i, it in enumerate(electrodes)
        }
        return Struct(electrodes, index, categories, channels, probability)

    def init_blueprint(self, blueprint: list[NpxElectrodeDesp]):
        for e in blueprint:
            self.categories[self.index[e.electrode]] = e.category

    def init_probability(self):
        for p in NpxProbeDesp.all_possible_categories().values():
            self.probability[self.categories == p] = category_mapping_probability(p)

    def selected_electrode(self) -> int:
        return np.count_nonzero(self.probability == 1)

    def add(self, e: int):
        self.probability[self.channels == self.channels[e]] = 0
        self.probability[e] = 1.0

    def get(self, e: int, c: int, r: int) -> int | None:
        eh, ec, er = self.electrodes[e].electrode
        return self.index.get((eh, ec + c, er + r), None)


def _select_loop(probe_type: ProbeType, s: Struct):
    while s.selected_electrode() < probe_type.n_channels:
        if (e := pick_electrode(s)) is not None:
            update_prob(s, e)
        else:
            break


def category_mapping_probability(p: int) -> float:
    match p:
        case NpxProbeDesp.CATE_SET:
            return 1.0
        case NpxProbeDesp.CATE_FULL:
            return 0.9
        case NpxProbeDesp.CATE_HALF:
            return 0.8  # 0.4, 0.2
        case NpxProbeDesp.CATE_QUARTER:
            return 0.7  # 0.35, 0.175
        case NpxProbeDesp.CATE_LOW:
            return 0.6
        case NpxProbeDesp.CATE_FORBIDDEN:
            return 0
        # case NpxProbeDesp.CATE_UNSET:
        case _:
            return 0.5


def build_channelmap(desp: NpxProbeDesp, chmap: ChannelMap, s: Struct) -> ChannelMap:
    ret = desp.new_channelmap(chmap)

    for e in np.nonzero(s.probability == 1)[0]:
        desp.add_electrode(ret, s.electrodes[e], overwrite=True)

    return ret


def information_entropy(s: Struct) -> float:
    p = s.probability[s.probability] > 0
    return -np.dot(p, np.log2(p))


def pick_electrode(s: Struct) -> int | None:
    mask = s.probability < 1
    cand = s.probability[mask]
    hp = np.max(cand)

    if hp == 0:
        return None

    return np.random.choice(np.arange(len(s.probability))[mask][cand >= hp])


def update_prob(s: Struct, e: int):
    s.add(e)
    r = np.array([it for it in surr(s, e) if it is not None])
    if len(r):
        r = r[s.probability[r] < 1]
        s.probability[r] /= 2


def surr(s: Struct, e: int) -> Iterator[int | None]:
    match int(s.categories[e]):
        case NpxProbeDesp.CATE_HALF:
            # o x o
            # x e x
            # o x o
            yield s.get(e, -1, 0)
            yield s.get(e, 1, 0)
            yield s.get(e, 0, 1)
            yield s.get(e, 0, -1)
        case NpxProbeDesp.CATE_QUARTER:
            # ? x ?
            # x x x
            # x e x
            # x x x
            # ? x ?
            yield s.get(e, -1, 0)
            yield s.get(e, 1, 0)
            yield s.get(e, -1, -1)
            yield s.get(e, 0, -1)
            yield s.get(e, 1, -1)
            yield s.get(e, -1, 1)
            yield s.get(e, 0, 1)
            yield s.get(e, 1, 1)
            yield s.get(e, 0, 2)
            yield s.get(e, 0, -2)
        case _:
            return
