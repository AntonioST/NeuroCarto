"""
Neuropixels another electrode selection method.
It has a *weaker* local density rule compared to the default one.
"""
import math
import random
from collections.abc import Iterator

from .desp import NpxProbeDesp, NpxElectrodeDesp, K
from .npx import ChannelMap, ProbeType

__all__ = ['electrode_select']


class E(NpxElectrodeDesp):
    prob: float


def electrode_select(desp: NpxProbeDesp, chmap: ChannelMap, blueprint: list[NpxElectrodeDesp],
                     **kwargs) -> ChannelMap:
    probe_type = chmap.probe_type

    cand: dict[K, E] = {
        it.electrode:
            E().copy(it, prob=0)
        for it in desp.all_electrodes(chmap)
    }

    for e in blueprint:
        cand[e.electrode].category = e.category

    for e in cand.values():
        e.prob = category_mapping_probability(e.category)

    for e in cand.values():
        # add pre-selected
        if e.category == NpxProbeDesp.CATE_SET:
            _add(desp, cand, e)

    _select_loop(desp, probe_type, cand)

    return build_channelmap(desp, chmap, cand)


def _select_loop(desp: NpxProbeDesp, probe_type: ProbeType, cand: dict[K, E]):
    while selected_electrode(cand) < probe_type.n_channels:
        if (e := pick_electrode(cand)) is not None:
            update_prob(desp, cand, e)
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


def selected_electrode(cond: dict[K, E]) -> int:
    return len([it for it in cond.values() if it.prob == 1])


def build_channelmap(desp: NpxProbeDesp, chmap: ChannelMap, cand: dict[K, E]) -> ChannelMap:
    ret = desp.new_channelmap(chmap)

    for e in cand.values():
        if e.prob == 1:
            desp.add_electrode(ret, e, overwrite=True)

    return ret


def information_entropy(cand: dict[K, E]) -> float:
    return -sum([it.prob * math.log2(it.prob) for it in cand.values() if it.prob > 0])


def pick_electrode(cand: dict[K, E]) -> E | None:
    if len(cand) == 0:
        raise RuntimeError()

    hp = max([it.prob for it in cand.values() if it.prob < 1])

    if hp == 0:
        return None

    sub = [it for it in cand.values() if hp <= it.prob < 1]
    assert len(sub) > 0
    return random.choice(sub)


def update_prob(desp: NpxProbeDesp, cand: dict[K, E], e: E):
    _add(desp, cand, e)

    for t in surr(cand, e):
        if t is not None and t.prob < 1:
            t.prob /= 2.0


def surr(cand: dict[K, E], e: E) -> Iterator[E | None]:
    match e.category:
        case NpxProbeDesp.CATE_HALF:
            # o x o
            # x e x
            # o x o
            yield _get(cand, e, -1, 0)
            yield _get(cand, e, 1, 0)
            yield _get(cand, e, 0, 1)
            yield _get(cand, e, 0, -1)
        case NpxProbeDesp.CATE_QUARTER:
            # ? x ?
            # x x x
            # x e x
            # x x x
            # ? x ?
            yield _get(cand, e, -1, 0)
            yield _get(cand, e, 1, 0)
            yield _get(cand, e, -1, -1)
            yield _get(cand, e, 0, -1)
            yield _get(cand, e, 1, -1)
            yield _get(cand, e, -1, 1)
            yield _get(cand, e, 0, 1)
            yield _get(cand, e, 1, 1)
            yield _get(cand, e, 0, 2)
            yield _get(cand, e, 0, -2)
        case _:
            return


def _add(desp: NpxProbeDesp, cand: dict[K, E], e: E):
    e.prob = 1.0

    for k in cand.values():
        if e.electrode != k.electrode and not desp.probe_rule(None, e, k):
            k.prob = 0


def _get(cand: dict[K, E], e: E, c: int, r: int) -> E | None:
    eh, ec, er = e.electrode
    return cand.get((eh, ec + c, er + r), None)
