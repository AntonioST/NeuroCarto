import textwrap
from collections.abc import Callable
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neurocarto.util.util_blueprint import BlueprintFunctions
from neurocarto.util.utils import doc_link
from .moving import move_i

__all__ = [
    'load_data', 'save_data',
    'interpolate_nan'
]


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.load_data.__doc__))
def load_data(self: BlueprintFunctions, file: str | Path) -> NDArray[np.float_]:
    """
    {DOC}
    :see: {BlueprintFunctions#load_data()}
    """
    e = self.probe.all_electrodes(self.channelmap)
    for t in e:
        t.category = np.nan
    e = self.probe.load_blueprint(file, e)
    return np.array([it.category for it in e], dtype=float)


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.save_data.__doc__))
def save_data(self: BlueprintFunctions, file: str | Path, value: NDArray[np.float_]):
    """
    {DOC}
    :see: {BlueprintFunctions#save_data()}
    """
    electrodes = self.apply_blueprint(blueprint=value.astype(int))
    np.save(file, self.probe.save_blueprint(electrodes))


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
