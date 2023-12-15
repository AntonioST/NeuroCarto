from __future__ import annotations

import abc
import sys
from pathlib import Path
from typing import Final

import numpy as np
from bokeh.models import ColumnDataSource, GlyphRenderer, UIElement
from bokeh.plotting import figure as Figure
from numpy.typing import NDArray

from chmap.config import ChannelMapEditorConfig
from chmap.views.base import ViewBase

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = ['DataReader', 'DataView']


class DataReader(metaclass=abc.ABCMeta):
    __match_args__ = 'filename',

    @classmethod
    def register(cls, filename: str, reader: type[DataReader]):
        """

        :param filename: filename pattern
        :param reader:
        :return:
        """
        pass

    @classmethod
    def load(cls, filename: str | Path) -> Self:
        pass

    filename: Final[Path]
    pos: NDArray[np.float_]  # Array[float, S, Y, (x, y)]
    data: NDArray[np.float_]  # Array[float, S, Y]

    def __init__(self, filename: Path):
        self.filename = filename

    @property
    def n_shank(self) -> int:
        if self.data.ndim == 2:
            return len(self.data)
        else:
            return 1


class DataView(ViewBase):
    data_reader: Final[DataReader]

    data_electrode: ColumnDataSource
    render_electrode: GlyphRenderer

    def __init__(self, config: ChannelMapEditorConfig, reader: DataReader):
        super().__init__(config)

        self.data_reader = reader
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

    def setup(self, **kwargs) -> list[UIElement]:
        pass

    # ================ #
    # updating methods #
    # ================ #

    def update(self):
        data = self.data_reader.pos

        x = []
        y = []
        for sh in range(data.shape[0]):
            x.append(list(data[sh, :, 0]))
            y.append(list(data[sh, :, 1]))

        self.data_electrode.data = dict(x=x, y=y)
