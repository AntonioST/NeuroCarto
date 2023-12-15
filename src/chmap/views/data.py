from __future__ import annotations

import abc
import sys
from pathlib import Path
from typing import Final

import numpy as np
from bokeh.models import ColumnDataSource, GlyphRenderer, UIElement
from bokeh.plotting import figure as Figure
from numpy.typing import NDArray

from chmap.util.utils import all_int

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
    pos: NDArray[np.float_]  # Array[float, Y]
    data: NDArray[np.float_]  # Array[float, [S,], Y]

    def __init__(self, filename: Path):
        self.filename = filename

    @property
    def n_shank(self) -> int:
        if self.data.ndim == 2:
            return len(self.data)
        else:
            return 1


class DataView:
    data_source: Final[list[DataReader]]

    data_electrode: ColumnDataSource
    render_electrode: GlyphRenderer

    def __init__(self):
        self.data_source = []
        self.data_electrode = ColumnDataSource(data=dict(x=[], y=[]))

    def setup(self) -> list[UIElement]:
        pass

    def plot(self, f: Figure):
        self.render_electrode = f.multi_line(
            'x', 'y', source=self.data_electrode
        )

    def get_data(self, file: str | Path) -> DataReader | None:
        file = Path(file).absolute()
        for data in self.data_source:
            if data.filename.absolute() == file:
                return data
        return None

    def get_data_index(self, file: str | Path | DataReader) -> int | None:
        if isinstance(file, DataReader):
            file = file.filename.absolute()
        else:
            file = Path(file).absolute()

        for i, data in enumerate(self.data_source):
            if data.filename.absolute() == file:
                return i
        return None

    def _index_render(self, i: int) -> int:
        ret = 0
        for j, data in enumerate(self.data_source):  # type: int, DataReader
            if j < i:
                ret += data.n_shank
            else:
                break
        return ret

    def add_data(self, data: str | Path | DataReader):
        if isinstance(data, (str, Path)):
            data = DataReader.load(data)
        if not isinstance(data, DataReader):
            raise TypeError()

        i = len(self.data_source)
        self.data_source.append(data)

        data = self.data_electrode.data
        x = data['x']
        y = data['y']

        for xx, yy in self._update_data(i):
            x.append(xx)
            y.append(yy)

        self.data_electrode.data = dict(x=x, y=y)

    def remove_data(self, data: int | str | Path | DataReader):
        if isinstance(data, (str, Path, DataReader)):
            if (data := self.get_data_index(data)) is None:
                return

        i = int(data)

        data = self.data_electrode.data
        x = data['x']
        y = data['y']

        k = self._index_render(i)
        n = self.data_source[i].n_shank

        del self.data_source[i]
        del x[k:k + n]
        del y[k:k + n]

        self.data_electrode.data = dict(x=x, y=y)

    def update_data(self, i: int | list[int] | None = None):
        match i:
            case None:
                i = set(range(len(self.data_source)))
            case i if all_int(i):
                i = {int(i)}
            case list():
                i = set(map(int, i))
            case _:
                raise TypeError()

        data = self.data_electrode.data
        ox = data['x']
        oy = data['y']

        x = []
        y = []
        for j in range(len(self.data_source)):
            if j in i:
                for xx, yy in self._update_data(j):
                    x.append(xx)
                    y.append(yy)
            else:
                k = len(x)
                for s in range(self.data_source[j].n_shank):
                    x.append(ox[s + k])
                    y.append(oy[s + k])

        self.data_electrode.data = dict(x=x, y=y)

    def _update_data(self, i: int) -> list[tuple[list[float], list[float]]]:
        data = self.data_source[i]
        y = list(data.pos)
        v = data.data
        if v.ndim == 1:
            return [(list(v), y)]
        elif v.ndim == 2:
            return [(list(v[:, it]), y) for it in range(v.shape[1])]
        else:
            raise ValueError()
