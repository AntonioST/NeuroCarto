import textwrap
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neurocarto.util.util_blueprint import BlueprintFunctions
from neurocarto.util.utils import doc_link
from .moving import move_i

__all__ = [
    'load_data',
    'load_csv_data',
    'load_date_from_blueprint_file',
    'save_data',
    'save_csv_data',
    'save_data_into_blueprint_file',
    'interpolate_nan'
]


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.load_data.__doc__))
def load_data(self: BlueprintFunctions, file: str | Path) -> NDArray[np.float_]:
    """
    {DOC}
    :see: {BlueprintFunctions#load_data()}
    :see: {load_csv_data()}
    :see: {load_date_from_blueprint()}
    """
    file = Path(file)
    match file.suffix:
        case '.npy':
            ret = np.load(file)
            if ret.shape == (len(self.s),) and np.issubdtype(ret.dtype, np.number):
                return ret.astype(float)
        case '.csv':
            return load_csv_data(self, file)
        case '.tsv':
            return load_csv_data(self, file, delimiter='\t')
        case _:
            return load_date_from_blueprint_file(self, file)


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.save_data.__doc__))
def save_data(self: BlueprintFunctions, file: str | Path, value: NDArray[np.float_]):
    """
    {DOC}
    :see: {BlueprintFunctions#save_data()}
    :see: {save_csv_data()}
    :see: {load_date_from_blueprint_file()}
    """
    file = Path(file)
    match file.suffix:
        case '.csv':
            save_csv_data(self, file, value)
        case '.tsv':
            save_csv_data(self, file, value, delimiter='\t')
        case _:
            save_data_into_blueprint_file(self, file, value)


def load_date_from_blueprint_file(self: BlueprintFunctions, file: str | Path):
    e = self.probe.all_electrodes(self.channelmap)
    for t in e:
        t.category = np.nan
    e = self.probe.load_blueprint(file, e)
    return np.array([it.category for it in e], dtype=float)


def save_data_into_blueprint_file(self: BlueprintFunctions, file: str | Path, value: NDArray[np.float_]):
    electrodes = self.apply_blueprint(blueprint=value.astype(int))
    np.save(file, self.probe.save_blueprint(electrodes))


@doc_link()
def load_csv_data(self: BlueprintFunctions, file: str | Path, *,
                  comments: str = '#',
                  delimiter: str = ',') -> NDArray[np.float_]:
    """
    Load electrode data array from a csv file.

    **Header**

    ``shank,x,y,value``

    where ``shank`` could also use ``s``,
    and ``value`` field can be arbitrary word.

    :param self:
    :param file: csv file.
    :param comments: default ``'#'``.
    :param delimiter: default ``','``.
    :return: Array[float, E] data array, where E is all electrodes.
    :raise IOError: unknown csv header.
    :raise: other error raised by ``np.loadtxt``.
    :see: {BlueprintFunctions#load_data()}
    """
    file = Path(file)

    with file.open() as f:
        for header_row, line in enumerate(f):
            if line.startswith(comments):
                continue

            header = line.strip()
            break

    match header.split(delimiter):
        case ['s' | 'shank', 'x', 'y', _]:
            pass
        case _:
            raise IOError(f'unknown header : {header}')

    data = np.loadtxt(file, dtype=float, comments=comments, delimiter=delimiter, skiprows=header_row)
    electrode = data[:, [0, 1, 2]].astype(int)
    value = data[:, 3]

    ret = np.full((len(self.s),), np.nan)
    ret[self.index_blueprint(electrode)] = value
    return ret


@doc_link()
def save_csv_data(self: BlueprintFunctions, file: str | Path, data: NDArray[np.float_], *,
                  comments: str = '#',
                  delimiter: str = ','):
    """
    Save electrode data into a csv file.

    **Header**

    ``shank,x,y,value``

    :param self:
    :param file: csv filepath
    :param data: Array[float, E] data array, where E is all electrodes.
    :param comments: default ``'#'``.
    :param delimiter: default ``','``.
    :raise: other error raised by ``np.savetxt``.
    """
    i = np.nonzero(~np.isnan(data))[0]
    save = np.column_stack([
        self.s[i],
        self.x[i],
        self.y[i],
        data[i],
    ])

    np.savetxt(file, save, delimiter=delimiter, comments=comments,
               header='shank,x,y,value',
               fmt=['%d', '%d', '%d', '%f'])


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.interpolate_nan.__doc__))
def interpolate_nan(self: BlueprintFunctions,
                    a: NDArray[np.float_],
                    kernel: int | tuple[int, int] = 1,
                    f: str | Callable[[NDArray[np.float_]], float] = 'mean') -> NDArray[np.float_]:
    """
    {DOC}
    :see: {BlueprintFunctions#interpolate_nan()}
    """
    if isinstance(f, str):
        if f == 'mean':
            f = np.nanmean
        elif f == 'median':
            f = np.nanmedian
        elif f == 'min':
            f = np.nanmin
        elif f == 'max':
            f = np.nanmax
        else:
            raise ValueError()

    if not np.any(m := np.isnan(a)):
        return a

    match kernel:
        case 0 | (0, 0):
            return a
        case int(y) if y > 0:
            kernel = (0, y)
        case (int(x), int(y)) if x >= 0 and y >= 0:
            pass
        case int() | (int(), int()):
            raise ValueError()
        case _:
            raise TypeError()

    r = []
    for tx in range(-kernel[0], kernel[0] + 1):
        for ty in range(-kernel[1], kernel[1] + 1):
            r.append(move_i(self, a, tx=tx, ty=ty, init=np.nan))

    r = f(r, axis=0)

    ret = a.copy()
    ret[m] = r[m]
    return ret
