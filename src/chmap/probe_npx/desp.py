from __future__ import annotations

import random
from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, TypeAlias

import numpy as np
from numpy.typing import NDArray

from chmap.probe import ProbeDesp, ElectrodeDesp, M, E
from chmap.probe_npx.npx import ChannelMap, Electrode, e2p, e2cb, ProbeType, ChannelHasUsedError

__all__ = ['NpxProbeDesp', 'NpxElectrodeDesp']

K: TypeAlias = tuple[int, int, int]


class NpxElectrodeDesp(ElectrodeDesp):
    electrode: K
    channel: int


class NpxProbeDesp(ProbeDesp[ChannelMap, NpxElectrodeDesp]):
    POLICY_D1: ClassVar = 11
    POLICY_D2: ClassVar = 12
    POLICY_D4: ClassVar = 13

    @property
    def possible_type(self) -> dict[str, int]:
        return {
            '4-Shank Neuropixels probe 2.0': 24,
            'Neuropixels probe 2.0': 21,
            'Neuropixels probe': 0,
        }

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
            'Set': self.POLICY_SET,
            #
            'Full Density': self.POLICY_D1,
            'Half Density': self.POLICY_D2,
            #
            'Quarter Density': self.POLICY_D4,
            'Sparse': self.POLICY_SPARSE,
            'Forbidden': self.POLICY_FORBIDDEN,
        }

    @property
    def channelmap_file_suffix(self) -> str:
        return '.imro'

    def load_from_file(self, file: Path) -> ChannelMap:
        return ChannelMap.from_imro(file)

    def save_to_file(self, chmap: ChannelMap, file: Path):
        chmap.save_imro(file)

    def new_channelmap(self, probe_type: int | ProbeType | ChannelMap = 24) -> ChannelMap:
        if isinstance(probe_type, ChannelMap):
            probe_type = probe_type.probe_type
        return ChannelMap(probe_type)

    def copy_channelmap(self, chmap: ChannelMap) -> ChannelMap:
        return ChannelMap(chmap.probe_type, chmap)

    def channelmap_desp(self, chmap: ChannelMap | None) -> str:
        if chmap is None:
            return '<b>Probe</b> 0/0'
        else:
            t = chmap.probe_type
            return f'<b>Probe[{t.code}]</b> {len(chmap)}/{t.n_channels}'

    def all_electrodes(self, chmap: ChannelMap) -> list[NpxElectrodeDesp]:
        probe_type = chmap.probe_type
        ret = []
        for s in range(probe_type.n_shank):
            for r in range(probe_type.n_row_shank):
                for c in range(probe_type.n_col_shank):
                    d = NpxElectrodeDesp()

                    d.electrode = (s, c, r)
                    d.x, d.y = e2p(probe_type, d.electrode)
                    d.channel, _ = e2cb(probe_type, d.electrode)

                    ret.append(d)
        return ret

    def all_channels(self, chmap: ChannelMap, s: Iterable[NpxElectrodeDesp] = None) -> list[NpxElectrodeDesp]:
        probe_type = chmap.probe_type
        ret = []
        for c, e in enumerate(chmap.channels):  # type: int, Electrode|None
            if e is not None:
                if s is None:
                    d = NpxElectrodeDesp()

                    d.electrode = (e.shank, e.column, e.row)
                    d.x, d.y = e2p(probe_type, e)
                    d.channel = c
                else:
                    d = self.get_electrode(s, (e.shank, e.column, e.row))

                if d is not None:
                    ret.append(d)

        return ret

    def is_valid(self, chmap: ChannelMap) -> bool:
        return len(chmap) == chmap.probe_type.n_channels

    def get_electrode(self, s: Iterable[NpxElectrodeDesp], e: K) -> NpxElectrodeDesp | None:
        return super().get_electrode(s, e)

    def add_electrode(self, chmap: ChannelMap, e: NpxElectrodeDesp, *, overwrite=False):
        try:
            chmap.add_electrode(e.electrode, exist_ok=True)
        except ChannelHasUsedError as x:
            if overwrite:
                chmap.del_electrode(x.electrode)
                chmap.add_electrode(e.electrode, exist_ok=True)

    def del_electrode(self, chmap: ChannelMap, e: NpxElectrodeDesp):
        chmap.del_electrode(e.electrode)

    def probe_rule(self, chmap: M, e1: E, e2: E) -> bool:
        return e1.channel != e2.channel

    def electrode_to_numpy(self, s: list[NpxElectrodeDesp]) -> NDArray[np.int_]:
        ret = np.zeros((len(s), 5), dtype=int)  # (N, (shank, col, row, state, policy))
        for i, e in enumerate(s):  # type: int, NpxElectrodeDesp
            h, c, r = e.electrode
            ret[i] = (h, c, r, e.state, e.policy)
        return ret

    def electrode_from_numpy(self, s: list[NpxElectrodeDesp], a: NDArray[np.int_]) -> list[NpxElectrodeDesp]:
        for data in a:
            e = (int(data[0]), int(data[1]), int(data[2]))
            state = int(data[3])
            policy = int(data[4])
            if (ee := self.get_electrode(s, e)) is not None:
                ee.state = state
                ee.policy = policy
        return s

    # ==================== #
    # electrode selections #
    # ==================== #

    def select_electrodes(self, chmap: ChannelMap, s: list[NpxElectrodeDesp], **kwargs) -> ChannelMap:
        ret = self.new_channelmap(chmap)

        cand: dict[K, NpxElectrodeDesp] = {it.electrode: it for it in self.all_electrodes(ret)}
        for e in s:
            cand[e.electrode].policy = e.policy

        for e in s:
            # add pre-selected
            if e.policy == self.POLICY_SET:
                self._add_electrode(ret, cand, e)

            # remove forbidden electrodes from the candidate set
            elif e.policy == self.POLICY_FORBIDDEN:
                try:
                    del cand[e.electrode]
                except KeyError:
                    pass

        return self._select_loop(ret, cand)

    def _select_loop(self, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], *, limit: int = 1000, **kwargs) -> ChannelMap:
        count = 0
        while len(cand) and count < limit:
            p, e = self._select_electrode(cand)
            if p == self.POLICY_FORBIDDEN:
                break
            elif e is not None:
                self._update(chmap, cand, e, p)
                count += 1
            else:
                break

        return chmap

    def _select_electrode(self, cand: dict[K, NpxElectrodeDesp]) -> tuple[int, NpxElectrodeDesp | None]:
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

    def _update(self, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp, policy: int):
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

    def _update_d1(self, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp):
        self._add_electrode(chmap, cand, e)

        if (t := _get(chmap, cand, e, 1, 0)) is not None:
            self._add_electrode(chmap, cand, t)

        if (t := _get(chmap, cand, e, 0, 1)) is not None:
            self._update_d1(chmap, cand, t)
        if (t := _get(chmap, cand, e, 0, -1)) is not None:
            self._update_d1(chmap, cand, t)

    def _update_d2(self, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp):
        self._add_electrode(chmap, cand, e)
        _del(cand, _get(chmap, cand, e, 1, 0))

        if (t := _get(chmap, cand, e, 1, 1)) is not None:
            self._update_d2(chmap, cand, t)
        if (t := _get(chmap, cand, e, 1, -1)) is not None:
            self._update_d2(chmap, cand, t)

    def _update_d4(self, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp):
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

    def _add_electrode(self, chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp):
        try:
            self.add_electrode(chmap, e)
        except BaseException:
            return

        _del(cand, e)

        for k in list(cand):
            if not self.probe_rule(chmap, e, cand[k]):
                del cand[k]


def _del(cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp | None):
    if e is not None:
        try:
            del cand[e.electrode]
        except KeyError:
            pass


def _get(chmap: ChannelMap, cand: dict[K, NpxElectrodeDesp], e: NpxElectrodeDesp, c: int, r: int) -> NpxElectrodeDesp | None:
    ret = cand.get(_move(chmap.probe_type, e, c, r), None)
    return ret if ret is not None and ret.policy == e.policy else None


def _move(probe_type: ProbeType, e: NpxElectrodeDesp, c: int, r: int) -> K:
    eh, ec, er = e.electrode
    nc = probe_type.n_col_shank
    return eh, (ec + c) % nc, (er + r)
