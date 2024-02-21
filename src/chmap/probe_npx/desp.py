from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, TypeAlias, Any

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
    CATE_FULL: ClassVar = 11  # full-density
    CATE_HALF: ClassVar = 12  # half-density
    CATE_QUARTER: ClassVar = 13  # quarter-density

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
    def possible_categories(self) -> dict[str, int]:
        return {
            'Unset': self.CATE_UNSET,
            'Set': self.CATE_SET,
            #
            'Full Density': self.CATE_FULL,
            'Half Density': self.CATE_HALF,
            #
            'Quarter Density': self.CATE_QUARTER,
            'Low priority': self.CATE_LOW,
            'Forbidden': self.CATE_FORBIDDEN,
        }

    def extra_controls(self, config: ChannelMapEditorConfig):
        from chmap.views.data_density import ElectrodeDensityDataView
        from chmap.views.view_efficient import ElectrodeEfficiencyData
        from .views import NpxReferenceControl
        return [NpxReferenceControl, ElectrodeDensityDataView, ElectrodeEfficiencyData]

    @property
    def channelmap_file_suffix(self) -> list[str]:
        return ['.imro', '.meta']

    def load_from_file(self, file: Path) -> ChannelMap:
        match file.suffix:
            case '.imro':
                return ChannelMap.from_imro(file)
            case '.meta':
                return ChannelMap.from_meta(file)
            case _:
                raise RuntimeError()

    def save_to_file(self, chmap: ChannelMap, file: Path):
        chmap.save_imro(file)

    def channelmap_code(self, chmap: Any | None) -> int | None:
        if not isinstance(chmap, ChannelMap):
            return None
        return chmap.probe_type.code

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

                    d.s = s
                    d.electrode = (s, c, r)
                    d.x, d.y = e2p(probe_type, d.electrode)
                    d.channel, _ = e2cb(probe_type, d.electrode)

                    ret.append(d)
        return ret

    def all_channels(self, chmap: ChannelMap, electrodes: Iterable[NpxElectrodeDesp] = None) -> list[NpxElectrodeDesp]:
        probe_type = chmap.probe_type
        ret = []
        for c, e in enumerate(chmap.channels):  # type: int, Electrode|None
            if e is not None:
                if electrodes is None:
                    d = NpxElectrodeDesp()

                    d.s = electrodes
                    d.electrode = (e.shank, e.column, e.row)
                    d.x, d.y = e2p(probe_type, e)
                    d.channel = c
                else:
                    d = self.get_electrode(electrodes, (e.shank, e.column, e.row))

                if d is not None:
                    ret.append(d)

        return ret

    def is_valid(self, chmap: ChannelMap) -> bool:
        return len(chmap) == chmap.probe_type.n_channels

    def get_electrode(self, electrodes: Iterable[NpxElectrodeDesp], e: K | NpxElectrodeDesp) -> NpxElectrodeDesp | None:
        return super().get_electrode(electrodes, e)

    def add_electrode(self, chmap: ChannelMap, e: NpxElectrodeDesp, *, overwrite=False):
        try:
            chmap.add_electrode(e.electrode, exist_ok=True)
        except ChannelHasUsedError as x:
            if overwrite:
                chmap.del_electrode(x.electrode)
                chmap.add_electrode(e.electrode, exist_ok=True)

    def del_electrode(self, chmap: ChannelMap, e: NpxElectrodeDesp):
        chmap.del_electrode(e.electrode)

    def clear_electrode(self, chmap: ChannelMap):
        del chmap.channels[:]

    def probe_rule(self, chmap: ChannelMap | None, e1: NpxElectrodeDesp, e2: NpxElectrodeDesp) -> bool:
        return e1.channel != e2.channel

    def save_blueprint(self, blueprint: list[NpxElectrodeDesp]) -> NDArray[np.int_]:
        ret = np.zeros((len(blueprint), 5), dtype=int)  # (N, (shank, col, row, state, category))
        for i, e in enumerate(blueprint):  # type: int, NpxElectrodeDesp
            s, c, r = e.electrode
            ret[i] = (s, c, r, e.state, e.category)
        return ret

    def load_blueprint(self, a: str | Path | NDArray[np.int_],
                       chmap: int | ProbeType | ChannelMap | list[NpxElectrodeDesp]) -> list[NpxElectrodeDesp]:
        if isinstance(a, (str, Path)):
            a = np.load(a)

        if isinstance(chmap, (int, ProbeType, ChannelMap)):
            electrodes = self.all_electrodes(chmap)
        elif isinstance(chmap, list):
            electrodes = chmap
        else:
            raise TypeError()

        c = {it.electrode: it for it in electrodes}
        for data in a:  # (shank, col, row, state, category)
            shank, col, row, state, category = data
            e = (int(shank), int(col), int(row))
            if (t := c.get(e, None)) is not None:
                t.state = int(state)
                t.category = int(category)

        return electrodes

    # ==================== #
    # electrode selections #
    # ==================== #

    def select_electrodes(self, chmap: ChannelMap, blueprint: list[NpxElectrodeDesp], *,
                          selector='default',
                          **kwargs) -> ChannelMap:
        from .select import electrode_select
        return electrode_select(self, chmap, blueprint, selector=selector, **kwargs)
