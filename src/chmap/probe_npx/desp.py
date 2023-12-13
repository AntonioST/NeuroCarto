from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from chmap.probe import ProbeDesp, ElectrodeDesp
from chmap.probe_npx.npx import ChannelMap, Electrode, e2p, e2cb

__all__ = ['NpxProbeDesp', 'NpxElectrodeDesp']


class NpxElectrodeDesp(ElectrodeDesp):
    electrode: tuple[int, int, int]
    channel: int


class NpxProbeDesp(ProbeDesp[ChannelMap, NpxElectrodeDesp]):

    def new_channelmap(self, probe_type: int | str) -> ChannelMap:
        return ChannelMap(probe_type)

    def copy_channelmap(self, chmap: ChannelMap) -> ChannelMap:
        return ChannelMap(chmap.probe_type, chmap)

    def all_electrodes(self, chmap: ChannelMap) -> list[NpxElectrodeDesp]:
        probe_type = chmap.probe_type
        ret = []
        for s in range(probe_type.n_shank):
            for r in range(probe_type.n_row_shank):
                for c in range(probe_type.n_col_shank):
                    d = NpxElectrodeDesp()

                    d.electrode = (s, c, r)
                    d.x, d.y = e2p(probe_type, d.electrode)
                    d.z = None
                    d.channel, _ = e2cb(probe_type, d.electrode)

                    ret.append(d)
        return ret

    def all_channels(self, chmap: ChannelMap) -> list[NpxElectrodeDesp]:
        probe_type = chmap.probe_type
        ret = []
        for c, e in enumerate(chmap.channels):  # type: int, Electrode|None
            if e is not None:
                d = NpxElectrodeDesp()

                d.electrode = (e.shank, e.column, e.row)
                d.x, d.y = e2p(probe_type, e)
                d.z = None
                d.channel = c

                ret.append(d)

        return ret

    def is_valid(self, chmap: ChannelMap) -> bool:
        return len(chmap) == chmap.probe_type.n_channels

    def get_electrode(self, s: list[NpxElectrodeDesp], e: tuple[int, int, int]) -> NpxElectrodeDesp | None:
        return super().get_electrode(s, e)

    def add_electrode(self, chmap: ChannelMap, e: NpxElectrodeDesp) -> ChannelMap:
        chmap.add_electrode(e.electrode)
        return chmap

    def del_electrode(self, chmap: ChannelMap, e: NpxElectrodeDesp) -> ChannelMap:
        chmap.del_electrode(e.electrode)
        return chmap

    def invalid_electrodes(self, chmap: ChannelMap, e: NpxElectrodeDesp, s: list[NpxElectrodeDesp]) -> list[NpxElectrodeDesp]:
        c = e.channel
        return [it for it in s if it.channel == c]

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
