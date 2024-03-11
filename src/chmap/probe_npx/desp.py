from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import ClassVar, TypeAlias, Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from chmap.config import ChannelMapEditorConfig
from chmap.probe import ProbeDesp, ElectrodeDesp
from chmap.probe_npx.npx import ChannelMap, Electrode, e2p, e2cb, ProbeType, ChannelHasUsedError, PROBE_TYPE
from chmap.util.utils import SPHINX_BUILD, doc_link

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from chmap.util.util_blueprint import BlueprintFunctions
    from chmap.views.blueprint import ProbePlotBlueprintCallback
elif SPHINX_BUILD:
    ProbeElectrodeDensityFunctor = 'chmap.views.data_density.ProbeElectrodeDensityFunctor'
    ProbePlotBlueprintFunctor = 'chmap.views.blueprint.ProbePlotBlueprintFunctor'
    ProbePlotElectrodeDataFunctor = 'chmap.views.blueprint_script.ProbePlotElectrodeDataFunctor'

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

    def new_channelmap(self, probe_type: int | str | ProbeType | ChannelMap = 24) -> ChannelMap:
        if isinstance(probe_type, (int, str)):
            probe_type = self.supported_type.get(probe_type, probe_type)
        elif isinstance(probe_type, ChannelMap):
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

        # Benchmark:
        #   run_script(profile)[optimize,sample_times=100,single_process=True]
        #       current             : 17.2320 seconds
        #           e2cb()            22.08%
        #       cache+copy()        : 23.4607 seconds
        #           copy()            50.39%
        #             dir()             17.10%
        #             str.startswith()  10.18%
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

                    d.s = e.shank
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

    def invalid_electrodes(self, chmap: ChannelMap, e: NpxElectrodeDesp | Iterable[NpxElectrodeDesp], electrodes: Iterable[NpxElectrodeDesp]) -> list[NpxElectrodeDesp]:
        if isinstance(e, Iterable):
            channels = set([it.channel for it in e])
            return [it for it in electrodes if it.channel in channels]
        else:
            return [it for it in electrodes if e.channel == it.channel]

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

    # ================== #
    # extension function #
    # ================== #

    def extra_controls(self, config: ChannelMapEditorConfig):
        from chmap.views.data_density import ElectrodeDensityDataView
        from chmap.views.view_efficient import ElectrodeEfficiencyData
        from .views import NpxReferenceControl
        return [NpxReferenceControl, ElectrodeDensityDataView, ElectrodeEfficiencyData]

    def view_ext_electrode_density(self, chmap: ChannelMap) -> NDArray[np.float_]:
        """

        :param chmap:
        :return:
        :see: {ProbeElectrodeDensityFunctor}
        """
        from .stat import npx_electrode_density
        return npx_electrode_density(chmap)

    def view_ext_statistics_info(self, bp: BlueprintFunctions) -> dict[str, str]:
        from .stat import npx_request_electrode, npx_channel_efficiency

        channelmap: ChannelMap = bp.channelmap
        used_channel = len(channelmap)
        used_channel_on_shanks = [
            len([it for it in channelmap.electrodes if it.shank == s])
            for s in range(channelmap.probe_type.n_shank)
        ]

        ucs = ', '.join(map(lambda it: f's{it[0]}={it[1]}', enumerate(used_channel_on_shanks)))

        electrodes = npx_request_electrode(bp)
        efficiency = npx_channel_efficiency(bp)

        return {
            'used channels': f'{used_channel}, total={channelmap.probe_type.n_channels}, ({ucs})',
            'request electrodes': f'{electrodes}',
            'channel efficiency': f'{100 * efficiency:.2f}%',
        }

    def view_ext_plot_blueprint(self, callback: ProbePlotBlueprintCallback, chmap: ChannelMap):
        """

        :param callback:
        :param chmap:
        :see: {ProbePlotBlueprintFunctor}
        """
        probe_type: ProbeType = chmap.probe_type
        c_space = probe_type.c_space
        r_space = probe_type.r_space
        size = c_space // 2, r_space // 2
        offset = c_space + c_space * probe_type.n_col_shank

        if 'Conflict' in callback.options:
            callback.blueprint = self._conflict_blueprint(callback.bp, callback.blueprint)
            callback.set_category_legend({'conflict': 'red'})
            callback.plot_blueprint({1: 'red'}, size, offset)
        else:
            categories = {
                self.CATE_FULL: 'green',
                self.CATE_HALF: 'orange',
                self.CATE_QUARTER: 'blue',
                self.CATE_FORBIDDEN: 'pink',
            }
            callback.blueprint = callback.bp.set(callback.blueprint, self.CATE_SET, self.CATE_FULL)
            callback.set_category_legend({
                'full-': 'green',
                'half-': 'orange',
                'quarter-': 'blue',
                'forbidden': 'pink',
            })
            callback.plot_blueprint(categories, size, offset)

    def _conflict_blueprint(self, bp: BlueprintFunctions, blueprint: NDArray[np.int_]) -> NDArray[np.int_]:
        i0 = bp.invalid(blueprint, electrodes=bp.channelmap, categories=[self.CATE_SET, self.CATE_FULL])
        r0 = bp.mask(blueprint, [self.CATE_FULL, self.CATE_HALF, self.CATE_QUARTER])
        c0 = i0 & r0

        i1 = bp.invalid(blueprint, categories=[self.CATE_SET, self.CATE_FULL])
        r1 = bp.mask(blueprint, [self.CATE_HALF, self.CATE_QUARTER])
        c1 = i1 & r1

        return (c0 | c1).astype(int)

    @doc_link(
        plot_category_area='chmap.probe_npx.plot.plot_category_area',
        plot_probe_shape='chmap.probe_npx.plot.plot_probe_shape',
    )
    def view_ext_blueprint_plot_categories(self, ax: Axes, chmap: ChannelMap, blueprint: NDArray[np.int_],
                                           color: dict[int, Any], *,
                                           probe_color: str | None = None,
                                           shank_width_scale=0.5,
                                           label_axis=False,
                                           **kwargs):
        """

        :param ax:
        :param chmap:
        :param blueprint:
        :param color:
        :param probe_color:
        :param shank_width_scale:
        :param label_axis:
        :param kwargs:
        :see: {ProbePlotElectrodeDataFunctor}, {plot_category_area()}, {plot_probe_shape()}
        """
        from .plot import plot_probe_shape, plot_category_area

        probe_type = chmap.probe_type

        plot_category_area(ax, probe_type, blueprint, electrode_unit='xyv', color=color, shank_width_scale=shank_width_scale, **kwargs)
        plot_probe_shape(ax, probe_type, color=probe_color, label_axis=label_axis, **kwargs)

        if not label_axis:
            ax.set_xlabel(None)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_ylabel(None)
            ax.set_yticks([])
            ax.set_yticklabels([])

    @doc_link(
        plot_electrode_block='chmap.probe_npx.plot.plot_electrode_block',
        plot_probe_shape='chmap.probe_npx.plot.plot_probe_shape',
    )
    def view_ext_blueprint_plot_electrode(self, ax: Axes, chmap: ChannelMap, data: NDArray[np.float_], *,
                                          probe_color: str | None = None,
                                          shank_width_scale=0.5,
                                          label_axis=False,
                                          **kwargs):
        """

        :param ax:
        :param chmap:
        :param data: Array[float, E], where E means all electrodes
        :param probe_color:
        :param shank_width_scale:
        :param label_axis:
        :param kwargs:
        :see: {ProbePlotElectrodeDataFunctor}, {plot_electrode_block()}, {plot_probe_shape()}
        """
        from .plot import plot_electrode_block, plot_probe_shape, electrode_coordinate
        probe_type = chmap.probe_type

        electrodes = electrode_coordinate(probe_type, electrode_unit='xy') / 1000
        data = np.vstack([electrodes.T, data]).T

        plot_electrode_block(ax, probe_type, data, electrode_unit='xyv', shank_width_scale=shank_width_scale, **kwargs)
        plot_probe_shape(ax, probe_type, color=probe_color, shank_width_scale=shank_width_scale, label_axis=label_axis, **kwargs)

        if not label_axis:
            ax.set_xlabel(None)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_ylabel(None)
            ax.set_yticks([])
            ax.set_yticklabels([])
