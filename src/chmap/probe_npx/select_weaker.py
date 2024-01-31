import math
import random
from collections.abc import Iterator

from .desp import NpxProbeDesp, NpxElectrodeDesp, K
from .npx import ChannelMap, ProbeType

__all__ = ['electrode_select']

DEBUG = False


class E(NpxElectrodeDesp):
    prob: float


def electrode_select(desp: NpxProbeDesp, chmap: ChannelMap, s: list[NpxElectrodeDesp],
                     **kwargs) -> ChannelMap:
    probe_type = chmap.probe_type
    ret = desp.new_channelmap(chmap)

    cand: dict[K, E] = {
        it.electrode:
            E().copy(it, prob=0)
        for it in desp.all_electrodes(ret)
    }

    for e in s:
        cand[e.electrode].policy = e.policy

    for e in cand.values():
        e.prob = policy_mapping_priority(e.policy)

    for e in cand.values():
        # add pre-selected
        if e.policy == NpxProbeDesp.POLICY_SET:
            _add(desp, cand, e)

    _select_loop(desp, probe_type, cand)

    return build_channelmap(desp, chmap, cand)


if not DEBUG:
    def _select_loop(desp: NpxProbeDesp, probe_type: ProbeType, cand: dict[K, E]):
        while selected_electrode(cand) < probe_type.n_channels:
            if (e := pick_electrode(cand)) is not None:
                update(desp, probe_type, cand, e)
            else:
                break
else:
    def _select_loop(desp: NpxProbeDesp, probe_type: ProbeType, cand: dict[K, E]):
        data = []
        count = 0
        try:
            while (n := selected_electrode(cand)) < probe_type.n_channels:
                if (e := pick_electrode(cand)) is not None:
                    p = e.prob
                    update(desp, probe_type, cand, e)
                    count += 1
                    data.append((n, policy_mapping_priority(e.policy), p, information_entropy(cand)))
                else:
                    break
        except KeyboardInterrupt:
            pass

        import numpy as np
        import matplotlib.pyplot as plt

        fg, ax = plt.subplots()
        data = np.array(data)
        data[:, 0] /= probe_type.n_channels
        data[:, 3] /= np.max(data[:, 3])
        ax.plot(data[:, 0], label='N')
        ax.plot(data[:, 1], label='Q')
        ax.plot(data[:, 2], label='P')
        ax.plot(data[:, 3], label='H')
        ax.legend()
        plt.show()


def policy_mapping_priority(p: int) -> float:
    match p:
        case NpxProbeDesp.POLICY_SET:
            return 1.0
        case NpxProbeDesp.POLICY_D1:
            return 0.9
        case NpxProbeDesp.POLICY_D2:
            return 0.8  # 0.4, 0.2
        case NpxProbeDesp.POLICY_D4:
            return 0.7  # 0.35, 0.125
        case NpxProbeDesp.POLICY_REMAINDER:
            return 0.6
        case NpxProbeDesp.POLICY_FORBIDDEN:
            return 0
        # case NpxProbeDesp.POLICY_UNSET:
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


def update(desp: NpxProbeDesp, probe_type: ProbeType, cand: dict[K, E], e: E):
    _add(desp, cand, e)

    for t in surr(probe_type, cand, e):
        if t is not None and t.prob < 1:
            t.prob /= 2.0


def surr(probe_type: ProbeType, cand: dict[K, E], e: E) -> Iterator[E | None]:
    match e.policy:
        case NpxProbeDesp.POLICY_D2:
            # x o
            # e x
            # x o
            yield _get(probe_type, cand, e, 1, 0)
            yield _get(probe_type, cand, e, 0, 1)
            yield _get(probe_type, cand, e, 0, -1)
        case NpxProbeDesp.POLICY_D4:
            # x o
            # x x
            # e x
            # x x
            # x o
            yield _get(probe_type, cand, e, 1, 0)
            yield _get(probe_type, cand, e, 0, 1)
            yield _get(probe_type, cand, e, 1, 1)
            yield _get(probe_type, cand, e, 0, -1)
            yield _get(probe_type, cand, e, 1, -1)
            yield _get(probe_type, cand, e, 0, 2)
            yield _get(probe_type, cand, e, 0, -2)
        case _:
            return


def _add(desp: NpxProbeDesp, cand: dict[K, E], e: E):
    e.prob = 1.0

    for k in cand.values():
        if e.electrode != k.electrode and not desp.probe_rule(None, e, k):
            k.prob = 0


def _get(probe_type: ProbeType, cand: dict[K, E], e: E, c: int, r: int) -> E | None:
    ret = cand.get(_move(probe_type, e, c, r), None)
    return ret if ret is not None and ret.policy == e.policy else None


def _move(probe_type: ProbeType, e: E, c: int, r: int) -> K:
    eh, ec, er = e.electrode
    nc = probe_type.n_col_shank
    return eh, (ec + c) % nc, (er + r)
