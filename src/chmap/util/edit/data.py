from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from chmap.util.util_blueprint import BlueprintFunctions
from .moving import move_i

__all__ = ['interpolate_nan']


def interpolate_nan(self: BlueprintFunctions,
                    a: NDArray[np.float_],
                    kernel: int | tuple[int, int] = 1,
                    f: str | Callable[[NDArray[np.float_]], float] = 'mean') -> NDArray[np.float_]:
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
