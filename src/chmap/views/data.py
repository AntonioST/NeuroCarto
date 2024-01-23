from __future__ import annotations

import abc

import numpy as np
from bokeh.models import ColumnDataSource, GlyphRenderer, UIElement, Div, FileInput
from bokeh.plotting import figure as Figure
from numpy.typing import NDArray

from chmap.config import ChannelMapEditorConfig
from chmap.probe import ProbeDesp, M, E
from chmap.util.utils import is_recursive_called
from chmap.views.base import ViewBase, DynamicView, InvisibleView

__all__ = ['DataView', 'Data1DView', 'FileDataView']


class DataView(ViewBase, InvisibleView, DynamicView, metaclass=abc.ABCMeta):
    """electrode data view base class."""

    data_electrode: ColumnDataSource | None = None
    render_electrode: GlyphRenderer | None = None

    def __init__(self, config: ChannelMapEditorConfig):
        super().__init__(config)

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """data name"""
        pass

    @abc.abstractmethod
    def data(self) -> dict | None:
        """get Electrode data. A dict used by ColumnDataSource."""
        pass

    # ================= #
    # render components #
    # ================= #

    # noinspection PyUnusedLocal
    def on_visible(self, visible: bool):
        if (render := self.render_electrode) is not None:
            render.visible = visible

    # ============= #
    # UI components #
    # ============= #

    def setup(self, **kwargs) -> list[UIElement]:
        from bokeh.layouts import row

        ret = [
            row(self.setup_visible_switch(), Div(text=f"<b>{self.name}</b>"))
        ]

        return ret

    # ================ #
    # updating methods #
    # ================ #

    def on_probe_update(self, probe: ProbeDesp[M, E], chmap: M | None, e: list[E] | None):
        self.update()

    def start(self):
        self.update()

    def update(self):
        """update the electrode data"""
        if (data := self.data()) is None:
            return

        if self.data_electrode is not None:
            self.data_electrode.data = data


class Data1DView(DataView, metaclass=abc.ABCMeta):
    """
    1D electrode data, represented in multi_line.
    """

    def __init__(self, config: ChannelMapEditorConfig):
        super().__init__(config)

        self.data_electrode = ColumnDataSource(data=dict(x=[], y=[]))

    @abc.abstractmethod
    def data(self) -> dict | None:
        """

        :return: dict(x=[[x]], y=[[y]])
        """
        pass

    # ================= #
    # render components #
    # ================= #

    def plot(self, f: Figure, **kwargs):
        self.render_electrode = f.multi_line('x', 'y', source=self.data_electrode, **kwargs)

    # ========= #
    # utilities #
    # ========= #

    @classmethod
    def arr_to_dict(cls, data: NDArray[np.float_]) -> dict:
        """

        :param data: Array[float, [S,], Y, (x, y)]
        :return: dict(x=[[x]], y=[[y]])
        """
        if data.ndim == 2:  # (Y, (x, y))
            xx = [list(data[:, 0])]
            yy = [list(data[:, 1])]
        elif data.ndim == 3:  # (S, Y, (x, y))
            xx = []
            yy = []
            for _data in data:
                xx.append(list(_data[:, 0]))
                yy.append(list(_data[:, 1]))
        else:
            raise RuntimeError(f'{type(set).__name__}.data() return .ndim{data.ndim}')

        return dict(x=xx, y=yy)


class FileDataView(DataView, metaclass=abc.ABCMeta):
    """Electrode data from a file."""
    
    @property
    @abc.abstractmethod
    def accept_file_ext(self) -> str:
        """

        https://docs.bokeh.org/en/latest/docs/reference/models/widgets/inputs.html#bokeh.models.FileInput.accept

        :return:
        """
        pass

    @abc.abstractmethod
    def load_data(self, filename: str):
        """Load electrode data from *filename*"""
        pass

    # ============= #
    # UI components #
    # ============= #

    data_input: FileInput

    def setup(self, **kwargs) -> list[UIElement]:
        ret = super().setup(**kwargs)

        self.data_input = FileInput(
            accept=self.accept_file_ext,
        )
        self.data_input.on_change('filename', self.on_data_selected)

        ret.append(self.data_input)

        return ret

    # noinspection PyUnusedLocal
    def on_data_selected(self, prop: str, old: str, filename: str):
        if is_recursive_called():
            return

        try:
            self.load_data(filename)
        except RuntimeError as e:
            from chmap.main_bokeh import ChannelMapEditorApp
            ChannelMapEditorApp.get_application().log_message(repr(e))
        else:
            if (probe := self._cache_probe) is not None:
                self.on_probe_update(probe, self._cache_channelmap, self._cache_electrodes)
            else:
                self.update()

    # ================ #
    # updating methods #
    # ================ #

    _cache_probe: ProbeDesp = None
    _cache_channelmap: M = None
    _cache_electrodes: list[E] = None

    def on_probe_update(self, probe: ProbeDesp[M, E], chmap: M | None, e: list[E] | None):
        self._cache_probe = probe
        self._cache_channelmap = chmap
        self._cache_electrodes = e

        self.update()
