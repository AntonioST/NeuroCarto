from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, TypeAlias

import numpy as np
from numpy.typing import NDArray

from chmap.config import ChannelMapEditorConfig
from chmap.probe import ProbeDesp, ElectrodeDesp
from chmap.probe_npx.npx import ChannelMap, Electrode, e2p, e2cb, ProbeType, ChannelHasUsedError, PROBE_TYPE

__all__ = ['NpxProbeDesp', 'NpxElectrodeDesp']

K: TypeAlias = tuple[int, int, int]


class NpxElectrodeDesp(ElectrodeDesp):
    electrode: K  # (shank, column, row)
    channel: int


class NpxProbeDesp(ProbeDesp[ChannelMap, NpxElectrodeDesp]):
    POLICY_D1: ClassVar = 11
    POLICY_D2: ClassVar = 12
    POLICY_D4: ClassVar = 13

    @property
    def supported_type(self) -> dict[str, int]:
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
    def possible_policies(self) -> dict[str, int]:
        return {
            'Unset': self.POLICY_UNSET,
            'Set': self.POLICY_SET,
            #
            'Full Density': self.POLICY_D1,
            'Half Density': self.POLICY_D2,
            #
            'Quarter Density': self.POLICY_D4,
            'Remainder': self.POLICY_REMAINDER,
            'Forbidden': self.POLICY_FORBIDDEN,
        }

    def extra_controls(self, config: ChannelMapEditorConfig):
        from chmap.views.data_density import ElectrodeDensityDataView
        from chmap.views.view_efficient import ElectrodeEfficiencyData
        from .views import NpxReferenceControl
        return [NpxReferenceControl, ElectrodeDensityDataView, ElectrodeEfficiencyData]

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
        return ChannelMap(chmap)

    def channelmap_desp(self, chmap: ChannelMap | None) -> str:
        if chmap is None:
            return '<b>Probe</b> 0/0'
        else:
            t = chmap.probe_type
            return f'<b>Probe[{t.code}]</b> {len(chmap)}/{t.n_channels}'

    def all_electrodes(self, chmap: int | ProbeType | ChannelMap) -> list[NpxElectrodeDesp]:
        if isinstance(chmap, int):
            probe_type = PROBE_TYPE[chmap]
        elif isinstance(chmap, ChannelMap):
            probe_type = chmap.probe_type
        elif isinstance(chmap, ProbeType):
            probe_type = chmap
        else:
            raise TypeError()

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

    def probe_rule(self, chmap: ChannelMap | None, e1: NpxElectrodeDesp, e2: NpxElectrodeDesp) -> bool:
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

    def select_electrodes(self, chmap: ChannelMap, s: list[NpxElectrodeDesp], *,
                          selector='default',
                          **kwargs) -> ChannelMap:
        from .select import electrode_select
        return electrode_select(self, chmap, s, selector=selector, **kwargs)
