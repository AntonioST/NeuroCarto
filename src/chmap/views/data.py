from __future__ import annotations

import abc
import sys
from typing import ClassVar, final

from bokeh.models import ColumnDataSource, GlyphRenderer, UIElement, FileInput, Div
from bokeh.plotting import figure as Figure

from chmap.config import ChannelMapEditorConfig
from chmap.probe import ProbeDesp, M, E
from chmap.util.utils import is_recursive_called
from chmap.views.base import ViewBase, DynamicView

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = ['DataReader', 'DataView']


class DataReader(metaclass=abc.ABCMeta):
    READERS: ClassVar[list[type[DataReader]]] = []

    def __init_subclass__(cls, **kwargs):
        cls.READERS.append(cls)

    def on_probe_update(self, probe: ProbeDesp[M, E], chmap: M | None, e: list[E] | None):
        pass

    @abc.abstractmethod
    def data(self) -> list[tuple[list[float], list[float]]] | None:
        """

        :return: [shank: ([x], [y])]
        """
        pass

    @classmethod
    def match_file(cls, filename: str) -> bool:
        raise NotImplementedError

    @classmethod
    @final
    def load_file(cls, filename: str) -> Self:
        for reader in cls.READERS:
            if reader.match_file(filename):
                return reader._load_file(filename)
        raise RuntimeError()

    @classmethod
    def _load_file(cls, filename: str) -> Self:
        raise NotImplementedError


class DataView(ViewBase, DynamicView):
    data_electrode: ColumnDataSource
    render_electrode: GlyphRenderer

    def __init__(self, config: ChannelMapEditorConfig, reader: DataReader = None):
        super().__init__(config)

        self.reader: DataReader | None = reader
        self.data_electrode = ColumnDataSource(data=dict(x=[], y=[]))

    # ================= #
    # render components #
    # ================= #

    def plot(self, f: Figure, **kwargs):
        self.render_electrode = f.multi_line(
            'x', 'y', source=self.data_electrode
        )

    # ============= #
    # UI components #
    # ============= #

    data_input: FileInput

    def setup(self, **kwargs) -> list[UIElement]:
        if self.reader is not None:
            return []

        self.data_input = FileInput()
        self.data_input.on_change('filename', self._on_data_selected)

        return [
            Div(text=f"<b>Data</b>"),
            self.data_input
        ]

    # noinspection PyUnusedLocal
    def _on_data_selected(self, prop: str, old: str, filename: str):
        if is_recursive_called():
            return

        try:
            self.reader = DataReader.load_file(filename)
        except RuntimeError as e:
            from chmap.main_bokeh import ChannelMapEditorApp
            ChannelMapEditorApp.get_application().log_message(repr(e))
        else:
            if isinstance(self.reader, DynamicView) and (probe := self._cache_probe) is not None:
                self.reader.on_probe_update(probe, self._cache_channelmap, self._cache_electrodes)

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

        self.reader.on_probe_update(probe, chmap, e)
        self.update()

    def update(self):
        if (reader := self.reader) is None:
            return

        if (data := reader.data()) is None:
            return

        xx = []
        yy = []
        for (x, y) in data:
            xx.append(list(x))
            yy.append(list(y))

        self.data_electrode.data = dict(x=xx, y=yy)
