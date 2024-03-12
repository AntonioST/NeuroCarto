from collections.abc import Iterator

import numpy as np

from neurocarto.util.util_blueprint import BlueprintFunctions

__all__ = ['surrounding', 'get_surrounding']


def surrounding(self: BlueprintFunctions, i: int | tuple[int, int, int], *, diagonal=True) -> Iterator[int]:
    if isinstance(i, (int, np.integer)):
        s = int(self.s[i])
        x = int(self.x[i] / self.dx)
        y = int(self.y[i] / self.dy)
    else:
        s, x, y = i
        s = int(s)
        x = int(x / self.dx)
        y = int(y / self.dy)

    if diagonal:
        code = [0, 1, 2, 3, 4, 5, 6, 7]
    else:
        code = [0, 2, 4, 6]

    pos = self._position_index
    for c in code:
        p = get_surrounding(self, (s, x, y), c)
        if (i := pos.get(p, None)) is not None:
            yield i


def get_surrounding(self: BlueprintFunctions, i: int | tuple[int, int, int], p: int) -> tuple[int, int, int]:
    # 3 2 1
    # 4 e 0
    # 5 6 7
    if isinstance(i, (int, np.integer)):
        s = int(self.s[i])
        x = int(self.x[i] / self.dx)
        y = int(self.y[i] / self.dy)
    else:
        s, x, y = i

    match p % 8:
        case 0:
            return s, x + 1, y
        case 1 | -7:
            return s, x + 1, y + 1
        case 2 | -6:
            return s, x, y + 1
        case 3 | -5:
            return s, x - 1, y + 1
        case 4 | -4:
            return s, x - 1, y
        case 5 | -3:
            return s, x - 1, y - 1
        case 6 | -2:
            return s, x, y - 1
        case 7 | -1:
            return s, x + 1, y - 1
        case _:
            raise RuntimeError('unreachable')
