"""
Neuropixels default electrode selection method.

"""
import random

from .desp import NpxProbeDesp, NpxElectrodeDesp, K
from .npx import ChannelMap, ProbeType

__all__ = ['electrode_select']


def electrode_select(desp: NpxProbeDesp, chmap: ChannelMap, blueprint: list[NpxElectrodeDesp], **kwargs) -> ChannelMap:
    ret = desp.new_channelmap(chmap)

    cand: dict[K, NpxElectrodeDesp] = {it.electrode: it for it in desp.all_electrodes(ret)}
    for e in blueprint:
        cand[e.electrode].category = e.category

    for e in blueprint:
        # add pre-selected
        if e.category == NpxProbeDesp.CATE_SET:
            _add(desp, ret, cand, e)

        # remove excluded electrodes from the candidate set
        elif e.category == NpxProbeDesp.CATE_EXCLUDED:
            try:
                del cand[e.electrode]
            except KeyError:
                pass

    return select_loop(desp, ret, cand, **kwargs)


def select_loop(desp: NpxProbeDesp, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp],
                **kwargs) -> ChannelMap:
    while len(cand):
        p, e = pick_electrode(cand)
        if p == NpxProbeDesp.CATE_EXCLUDED:
            break
        elif e is not None:
            update(desp, chmap, cand, e, p)
        else:
            break

    return chmap


def pick_electrode(cand: dict[K, NpxElectrodeDesp]) -> tuple[int, NpxElectrodeDesp | None]:
    if len(cand) == 0:
        return NpxProbeDesp.CATE_EXCLUDED, None

    if len(ret := [e for e in cand.values() if e.category == NpxProbeDesp.CATE_FULL]) > 0:
        return NpxProbeDesp.CATE_FULL, random.choice(ret)

    if len(ret := [e for e in cand.values() if e.category == NpxProbeDesp.CATE_HALF]) > 0:
        return NpxProbeDesp.CATE_HALF, random.choice(ret)

    if len(ret := [e for e in cand.values() if e.category == NpxProbeDesp.CATE_QUARTER]) > 0:
        return NpxProbeDesp.CATE_QUARTER, random.choice(ret)

    if len(ret := [e for e in cand.values() if e.category == NpxProbeDesp.CATE_LOW]) > 0:
        return NpxProbeDesp.CATE_LOW, random.choice(ret)

    if len(ret := [e for e in cand.values() if e.category == NpxProbeDesp.CATE_UNSET]) > 0:
        return NpxProbeDesp.CATE_UNSET, random.choice(ret)

    return NpxProbeDesp.CATE_EXCLUDED, None


def update(desp: NpxProbeDesp, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp, category: int):
    match category:
        case NpxProbeDesp.CATE_FULL:
            return update_d1(desp, chmap, cand, e)
        case NpxProbeDesp.CATE_HALF:
            return update_d2(desp, chmap, cand, e)
        case NpxProbeDesp.CATE_QUARTER:
            return update_d4(desp, chmap, cand, e)
        case NpxProbeDesp.CATE_LOW | NpxProbeDesp.CATE_UNSET:
            return _add(desp, chmap, cand, e)
        case _:
            raise ValueError()


def update_d1(desp: NpxProbeDesp, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp):
    _add(desp, chmap, cand, e)

    if (t := _get(chmap, cand, e, 1, 0)) is not None:
        _add(desp, chmap, cand, t)

    if (t := _get(chmap, cand, e, 0, 1)) is not None:
        update_d1(desp, chmap, cand, t)
    if (t := _get(chmap, cand, e, 0, -1)) is not None:
        update_d1(desp, chmap, cand, t)


def update_d2(desp: NpxProbeDesp, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp):
    _add(desp, chmap, cand, e)
    _del(cand, _get(chmap, cand, e, 1, 0))
    _del(cand, _get(chmap, cand, e, 0, 1))
    _del(cand, _get(chmap, cand, e, 0, -1))

    if (t := _get(chmap, cand, e, 1, 1)) is not None:
        update_d2(desp, chmap, cand, t)
    if (t := _get(chmap, cand, e, 1, -1)) is not None:
        update_d2(desp, chmap, cand, t)


def update_d4(desp: NpxProbeDesp, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp):
    _add(desp, chmap, cand, e)
    _del(cand, _get(chmap, cand, e, 1, 0))
    _del(cand, _get(chmap, cand, e, 0, 1))
    _del(cand, _get(chmap, cand, e, 1, 1))
    _del(cand, _get(chmap, cand, e, 0, -1))
    _del(cand, _get(chmap, cand, e, 1, -1))
    _del(cand, _get(chmap, cand, e, 0, 2))
    _del(cand, _get(chmap, cand, e, 0, -2))

    if (t := _get(chmap, cand, e, 1, 2)) is not None:
        update_d4(desp, chmap, cand, t)
    if (t := _get(chmap, cand, e, 1, -2)) is not None:
        update_d4(desp, chmap, cand, t)


def _add(desp: NpxProbeDesp, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp):
    try:
        desp.add_electrode(chmap, e)
    except BaseException:
        return

    _del(cand, e)

    for k in list(cand):
        # if not desp.probe_rule(chmap, e, cand[k]):
        if e.channel == cand[k].channel:
            del cand[k]


def _del(cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp | None):
    if e is not None:
        try:
            del cand[e.electrode]
        except KeyError:
            pass


def _get(chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp, c: int, r: int) -> NpxElectrodeDesp | None:
    ret = cand.get(_move(chmap.probe_type, e, c, r), None)
    return ret if ret is not None and ret.category == e.category else None


def _move(probe_type: ProbeType, e: NpxElectrodeDesp, c: int, r: int) -> K:
    eh, ec, er = e.electrode
    nc = probe_type.n_col_shank
    return eh, (ec + c) % nc, (er + r)
