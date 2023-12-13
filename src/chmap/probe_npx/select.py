import random
from typing import ClassVar, TypeAlias

from chmap.select import Selector
from .desp import NpxProbeDesp, NpxElectrodeDesp
from .npx import ChannelMap, ProbeType

__all__ = ['NpxSelector']

E: TypeAlias = tuple[int, int, int]


class NpxSelector(Selector[ChannelMap, NpxElectrodeDesp]):
    POLICY_D1: ClassVar = 21
    POLICY_D2: ClassVar = 22
    POLICY_D4: ClassVar = 23

    desp: NpxProbeDesp

    def __init__(self, desp: NpxProbeDesp):
        super().__init__(desp)

    @property
    def possible_states(self) -> dict[str, int]:
        return {
            'Enable': self.STATE_USED,
            'Disable': self.STATE_UNUSED
        }

    @property
    def possible_policy(self) -> dict[str, int]:
        return {
            'Unset': self.POLICY_UNSET,
            'Forbidden': self.POLICY_FORBIDDEN,
            'Sparse': self.POLICY_SPARSE,
            'Full Density': self.POLICY_D1,
            'Half Density': self.POLICY_D2,
            'Quarter Density': self.POLICY_D4,
        }

    def run(self, chmap: ChannelMap, s: list[NpxElectrodeDesp]) -> ChannelMap:
        ret = self.desp.new_channelmap(chmap)
        s = self.desp.copy_electrode(s)

        # add pre-selected electrodes
        selected = []
        cand: dict[E, NpxElectrodeDesp] = {}
        for e in s:
            if e.policy == self.STATE_USED:
                selected.append(e)
            else:
                cand[e.electrode] = e

        # remove selected electrodes from the candidate set
        for e in selected:
            self._add_electrode(ret, cand, e)

        return self._run_loop(ret, cand)

    def _run_loop(self, chmap: ChannelMap, cand: dict[E, NpxElectrodeDesp], *, limit: int = 1000) -> ChannelMap:
        # remove forbidden electrodes from the candidate set
        cand = {k: e for k, e in cand.items() if e.policy != self.POLICY_FORBIDDEN}

        count = 0
        while len(cand) and count < limit:
            p, e = self._select(cand)
            if p == self.POLICY_FORBIDDEN:
                break
            elif e is not None:
                self._update(chmap, cand, e, p)
                count += 1
            else:
                break

        return chmap

    def _select(self, cand: dict[E, NpxElectrodeDesp]) -> tuple[int, NpxElectrodeDesp | None]:
        if len(cand) == 0:
            return self.POLICY_FORBIDDEN, None

        if len(ret := [e for e in cand.values() if e.policy == self.POLICY_D1]) > 0:
            return self.POLICY_D1, random.choice(ret)

        if len(ret := [e for e in cand.values() if e.policy == self.POLICY_D2]) > 0:
            return self.POLICY_D2, random.choice(ret)

        if len(ret := [e for e in cand.values() if e.policy == self.POLICY_D4]) > 0:
            return self.POLICY_D4, random.choice(ret)

        if len(ret := [e for e in cand.values() if e.policy == self.POLICY_SPARSE]) > 0:
            return self.POLICY_SPARSE, random.choice(ret)

        if len(ret := [e for e in cand.values() if e.policy == self.POLICY_UNSET]) > 0:
            return self.POLICY_UNSET, random.choice(ret)

        return self.POLICY_FORBIDDEN, None

    def _update(self, chmap: ChannelMap, cand: dict[E, NpxElectrodeDesp], e: NpxElectrodeDesp, policy: int):
        match policy:
            case self.POLICY_D1:
                return self._update_d1(chmap, cand, e)
            case self.POLICY_D2:
                return self._update_d2(chmap, cand, e)
            case self.POLICY_D4:
                return self._update_d4(chmap, cand, e)
            case self.POLICY_SPARSE | self.POLICY_UNSET:
                return self._add_electrode(chmap, cand, e)
            case _:
                raise ValueError()

    def _update_d1(self, chmap: ChannelMap, cand: dict[E, NpxElectrodeDesp], e: NpxElectrodeDesp):
        self._add_electrode(chmap, cand, e)

        if (t := _get(chmap, cand, e, 1, 0)) is not None:
            self._add_electrode(chmap, cand, t)

        if (t := _get(chmap, cand, e, 0, 1)) is not None:
            self._update_d1(chmap, cand, t)
        if (t := _get(chmap, cand, e, 0, -1)) is not None:
            self._update_d1(chmap, cand, t)

    def _update_d2(self, chmap: ChannelMap, cand: dict[E, NpxElectrodeDesp], e: NpxElectrodeDesp):
        self._add_electrode(chmap, cand, e)
        _del(cand, _get(chmap, cand, e, 1, 0))

        if (t := _get(chmap, cand, e, 1, 1)) is not None:
            self._update_d2(chmap, cand, t)
        if (t := _get(chmap, cand, e, 1, -1)) is not None:
            self._update_d2(chmap, cand, t)

    def _update_d4(self, chmap: ChannelMap, cand: dict[E, NpxElectrodeDesp], e: NpxElectrodeDesp):
        self._add_electrode(chmap, cand, e)
        _del(cand, _get(chmap, cand, e, 1, 0))
        _del(cand, _get(chmap, cand, e, 0, 1))
        _del(cand, _get(chmap, cand, e, 1, 1))
        _del(cand, _get(chmap, cand, e, 0, -1))
        _del(cand, _get(chmap, cand, e, 1, -1))

        if (t := _get(chmap, cand, e, 1, 2)) is not None:
            self._update_d4(chmap, cand, t)
        if (t := _get(chmap, cand, e, 1, -2)) is not None:
            self._update_d4(chmap, cand, t)

    def _add_electrode(self, chmap: ChannelMap, cand: dict[E, NpxElectrodeDesp], e: NpxElectrodeDesp):
        try:
            self.desp.add_electrode(chmap, e)
        except BaseException:
            return

        _del(cand, e)

        for k in list(cand):
            if not self.desp.probe_rule(chmap, e, cand[k]):
                del cand[k]


def _del(cand: dict[E, NpxElectrodeDesp], e: NpxElectrodeDesp | None):
    if e is not None:
        try:
            del cand[e.electrode]
        except KeyError:
            pass


def _get(chmap: ChannelMap, cand: dict[E, NpxElectrodeDesp], e: NpxElectrodeDesp, c: int, r: int) -> NpxElectrodeDesp | None:
    ret = cand.get(_move(chmap.probe_type, e, c, r))
    return ret if ret is not None and ret.policy == e.policy else None


def _move(probe_type: ProbeType, e: NpxElectrodeDesp, c: int, r: int) -> E:
    eh, ec, er = e.electrode
    nc = probe_type.n_col_shank
    return eh, (ec + c) % nc, (er + r)
