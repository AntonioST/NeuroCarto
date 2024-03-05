from numpy.typing import NDArray

from chmap.util.util_blueprint import BlueprintFunctions

__all__ = ['print_local']


def print_local(self: BlueprintFunctions, data: NDArray, i: int, size: int = 1) -> str:
    """
    print electrode data around the electrode *i*.

    :param self:
    :param data: electrode data
    :param i: electrode index
    :param size: local size
    :return: ascii art text.
    """
    s = int(self.s[i])
    x = int(self.x[i] / self.dx)
    y = int(self.y[i] / self.dy)

    ret = []
    for dy in range(-size, size + 1):
        ret.append((row := []))
        for dx in range(-size, size + 1):
            j = self._position_index.get((s, x + dx, y + dy), None)
            if j is None:
                row.append('_')
            else:
                row.append(str(data[j]))

    width = max([max([len(it) for it in row]) for row in ret])
    fmt = f'%{width}s'
    return '\n'.join(reversed([
        ' '.join([
            fmt % it
            for it in row
        ]) for row in ret
    ]))
