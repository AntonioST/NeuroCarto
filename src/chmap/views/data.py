from __future__ import annotations

import abc
import logging
from pathlib import Path

import numpy as np
from bokeh.models import ColumnDataSource, GlyphRenderer, UIElement
from bokeh.plotting import figure as Figure
from numpy.typing import NDArray

from chmap.config import ChannelMapEditorConfig
from chmap.probe import ProbeDesp, M, E
from chmap.util.bokeh_app import run_later
from chmap.util.bokeh_util import is_recursive_called, PathAutocompleteInput
from chmap.views.base import ViewBase, DynamicView, InvisibleView

__all__ = ['DataView', 'DataHandler', 'Data1DView', 'FileDataView']


class DataHandler:
    """A data receiver view."""

    def on_data_update(self, probe: ProbeDesp[M, E], e: list[E], data: NDArray[np.float_] | None):
        """

        :param probe:
        :param e: N-length electrodes
        :param data: Array[float, N] electrode data.
        """
        pass


class DataView(ViewBase, InvisibleView, DynamicView, metaclass=abc.ABCMeta):
    """electrode data view base class."""

    logger: logging.Logger | None = None
    data_electrode: ColumnDataSource | None = None
    render_electrode: GlyphRenderer | None = None

    @abc.abstractmethod
    def data(self) -> dict | None:
        """get Electrode data. A dict used by ColumnDataSource."""
        pass

    # ================ #
    # updating methods #
    # ================ #

    def on_probe_update(self, probe: ProbeDesp[M, E], chmap: M | None, electrodes: list[E] | None):
        run_later(self.update)

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

    def __init__(self, config: ChannelMapEditorConfig, **kwargs):
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

    @classmethod
    def arr_to_dict(cls, data: NDArray[np.float_]) -> dict:
        """

        :param data: Array[float, [S,], Y, (x, y)]
        :return: dict(x=[array[x]], y=[array[y]])
        """
        xx: list[NDArray[np.float_]]
        yy: list[NDArray[np.float_]]

        if data.ndim == 2:  # (Y, (x, y))
            xx = [data[:, 0]]
            yy = [data[:, 1]]
        elif data.ndim == 3:  # (S, Y, (x, y))
            xx = []
            yy = []
            for _data in data:
                xx.append(_data[:, 0])
                yy.append(_data[:, 1])
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
    def on_data_selected(self, filename: Path):
        if is_recursive_called():
            return

        try:
            self.load_data(filename)
        except RuntimeError as e:
            if (logger := self.logger) is not None:
                logger.warning('load_data() fail', exc_info=e)
        else:
            run_later(self.update)
