from __future__ import annotations

import abc
from pathlib import Path
from typing import Any

import numpy as np
from bokeh.models import ColumnDataSource, GlyphRenderer, UIElement
from numpy.typing import NDArray

from neurocarto.config import CartoConfig
from neurocarto.probe import ProbeDesp, ElectrodeDesp
from neurocarto.util.bokeh_app import run_later
from neurocarto.util.bokeh_util import is_recursive_called, PathAutocompleteInput
from neurocarto.util.util_blueprint import BlueprintFunctions
from neurocarto.views.base import Figure, ViewBase, DynamicView, InvisibleView

__all__ = ['DataView', 'Data1DView', 'FileDataView']


class DataView(ViewBase, InvisibleView, DynamicView, metaclass=abc.ABCMeta):
    """electrode data view base class."""

    data_electrode: ColumnDataSource | None = None
    render_electrode: GlyphRenderer | None = None

    @abc.abstractmethod
    def data(self) -> dict | None:
        """get Electrode data. A dict used by ColumnDataSource."""
        pass

    # ============== #
    # probe updating #
    # ============== #

    probe: ProbeDesp | None = None
    channelmap: Any | None = None
    blueprint: list[ElectrodeDesp] | None = None

    def on_probe_update(self, probe, chmap, electrodes):
        self.probe = probe
        self.channelmap = chmap
        self.blueprint = electrodes

        run_later(self.update)

    def new_blueprint_function(self) -> BlueprintFunctions:
        """

        :return:
        :raise RuntimeError: no probe existed.
        """
        if (probe := self.probe) is not None:
            return BlueprintFunctions(probe, self.channelmap)
        else:
            raise RuntimeError()

    # ================ #
    # updating methods #
    # ================ #

    def start(self):
        self.update()

    def update(self):
        """update the electrode data"""
        from neurocarto.util.debug import Profiler

        with Profiler('.neurocarto.profile-data.dat', enable='NEUROCARTO_PROFILE_VIEW_DATA') as profile:
            data = self.data()

        if profile.enable:
            self.logger.debug('data() used %.2f sec', profile.duration)
            profile.print_command()

        if data is None:
            return

        if self.data_electrode is not None:
            self.data_electrode.data = data


class Data1DView(DataView, metaclass=abc.ABCMeta):
    """
    1D electrode data, represented in multi_line.
    """

    def __init__(self, config: CartoConfig, **kwargs):
        super().__init__(config, **kwargs)

        self.data_electrode = ColumnDataSource(data=dict(x=[], y=[]))

    @abc.abstractmethod
    def data(self) -> dict | None:
        """

        :return: dict(x=[[x]], y=[[y]])
        """
        pass

    # ============= #
    # UI components #
    # ============= #

    def _setup_render(self, f: Figure, **kwargs):
        self.render_electrode = f.multi_line('x', 'y', source=self.data_electrode, **kwargs)

    # ========= #
    # utilities #
    # ========= #

    _cache_channelmap_code: int | None = None
    _cache_shank_space: list[tuple[float, float]] | None = None

    def transform(self, data: NDArray[np.float_], height: float = 1, vmax: float = None) -> NDArray[np.float_]:
        """
        normalize and transform 2D value array to 2D curve array.

        :param data: Array[float, [S,], (v, y), Y]
        :param height: ratio of max(data) / shank_space
        :param vmax:
        :return: Array[float, [S,], (x, y), Y]
        """
        if self.probe is None:
            raise RuntimeError('missing probe')

        if (channelmap_code := self.probe.channelmap_code(self.channelmap)) is None:
            raise RuntimeError('missing probe')

        # check first time or probe type changed.
        if self._cache_shank_space is None or self._cache_channelmap_code != channelmap_code:
            self._update_shank_space()
            if self._cache_shank_space is None:
                raise RuntimeError('missing probe')
            self._cache_channelmap_code = channelmap_code

        #
        if data.ndim == 2:  # ((v,y), Y)
            if vmax is None:
                vmax = np.nanmax(data[0], initial=0)

            x = self._cache_shank_space[0]
            if vmax == 0:
                return np.vstack([np.full_like(data[0], x[0]), data[1]])
            else:
                return np.vstack([x[0] + data[0] * height * x[1] / vmax, data[1]])

        elif data.ndim == 3:  # (S,(v,y),Y)
            if vmax is None:
                vmax = np.nanmax(data[:, 0, :], initial=0)

            if vmax == 0:
                return np.array([
                    np.vstack([np.full_like(_data[0], x[0]), _data[1]])
                    for i, _data in enumerate(data)
                    if (x := self._cache_shank_space[i])
                ])
            else:
                return np.array([
                    np.vstack([x[0] + _data[0] * height * x[1] / vmax, _data[1]])
                    for i, _data in enumerate(data)
                    if (x := self._cache_shank_space[i])
                ])
        else:
            raise RuntimeError()

    def _update_shank_space(self):
        bp = self.new_blueprint_function()
        if bp.channelmap is None:
            self._cache_shank_space = None
            return

        shank_space = []
        x0 = 0
        xs = 200
        for i, s in enumerate(np.unique(bp.s)):
            x = bp.x[bp.s == s]

            x2 = np.min(x)
            x3 = np.max(x)

            if i > 0:
                xs = float(x2 - x0)
                shank_space.append((float(x0), xs))

            x0 = x3

        shank_space.append((float(x0), xs))
        self._cache_shank_space = shank_space

    @classmethod
    def arr_to_dict(cls, data: NDArray[np.float_]) -> dict:
        """

        :param data: Array[float, [S,], (x, y), Y]
        :return: dict(x=[array[x]], y=[array[y]])
        """
        xx: list[NDArray[np.float_]]
        yy: list[NDArray[np.float_]]

        if data.ndim == 2:  # ((x,y),Y)
            xx = [data[0]]
            yy = [data[1]]
        elif data.ndim == 3:  # (S,(x,y),Y)
            xx = []
            yy = []
            for _data in data:
                xx.append(_data[0])
                yy.append(_data[1])
        else:
            raise RuntimeError(f'{type(set).__name__}.data() return .ndim{data.ndim}')

        return dict(x=xx, y=yy)


class FileDataView(DataView, metaclass=abc.ABCMeta):
    """Electrode data from a file."""

    @property
    def name(self) -> str:
        return 'Data Path'

    @abc.abstractmethod
    def load_data(self, filename: Path):
        """Load electrode data from *filename*"""
        pass

    # ============= #
    # UI components #
    # ============= #

    data_input: PathAutocompleteInput

    def setup_data_input(self, root: Path = None,
                         accept: list[str] = None,
                         width=300,
                         **kwargs) -> PathAutocompleteInput:
        if root is None:
            root = Path('.')

        self.data_input = PathAutocompleteInput(
            root,
            self.on_data_selected,
            mode='file',
            accept=accept,
            width=width,
            **kwargs
        )
        return self.data_input

    def _setup_title(self, **kwargs) -> list[UIElement]:
        ret = super()._setup_title(**kwargs)

        self.view_title.text = '<b>Data Path</b>'
        data_input = self.setup_data_input()
        ret.insert(-1, data_input.input)

        return ret

    # noinspection PyUnusedLocal
    def on_data_selected(self, filename: Path | None):
        if is_recursive_called() or filename is None:
            return

        try:
            self.load_data(filename)
        except RuntimeError as e:
            if (logger := self.logger) is not None:
                logger.warning('load_data() fail', exc_info=e)
        else:
            run_later(self.update)
