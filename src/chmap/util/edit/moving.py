import numpy as np
from numpy.typing import NDArray

from chmap.util.util_blueprint import BlueprintFunctions

__all__ = ['move', 'move_i', 'fill', 'extend']


def move(self: BlueprintFunctions, a: NDArray, *,
         tx: int = 0, ty: int = 0,
         shanks: list[int] = None,
         axis: int = 0,
         init: float = 0) -> NDArray:
    s = self.s
    x = self.x
    y = self.y
    dx = self.dx
    dy = self.dy

    if a.shape[axis] != len(s):
        raise RuntimeError()

    if abs(tx) < dx and abs(ty) < dy:
        return a

    pos = self._position_index

    ii = []
    jj = []
    for i in range(len(s)):
        if shanks is None or s[i] in shanks:
            p = int(s[i]), int((x[i] + tx) / dx), int((y[i] + ty) / dy)
            j = pos.get(p, None)
        else:
            j = i

        if j is not None:
            ii.append(i)
            jj.append(j)

    ii = np.array(ii)
    jj = np.array(jj)

    if a.ndim > 1:
        _index = [slice(None)] * a.ndim
        _index[axis] = ii
        ii = tuple(_index)
        _index[axis] = jj
        jj = tuple(_index)

    ret = np.full_like(a, init)
    ret[jj] = a[ii]

    return ret


def move_i(self: BlueprintFunctions, a: NDArray, *,
           tx: int = 0, ty: int = 0,
           shanks: list[int] = None,
           axis: int = 0,
           init: float = 0) -> NDArray:
    if tx == 0 and ty == 0:
        return a
    return move(self, a, tx=tx * self.dx, ty=ty * self.dy, shanks=shanks, axis=axis, init=init)


def fill(self: BlueprintFunctions,
         blueprint: NDArray[np.int_],
         categories: int | list[int] = None, *,
         threshold: int = None,
         gap: int | None = 1,
         unset: bool = False) -> NDArray[np.int_]:
    if len(blueprint) != len(self.s):
        raise ValueError()

    if gap is None:
        gap_window = None
    else:
        if gap <= 0:
            raise ValueError()
        gap_window = gap + 2

    if categories is None:
        categories = list(set(self.categories.values()))
    elif isinstance(categories, int):
        categories = [categories]

    dx = self.dx
    dy = self.dy

    ret = blueprint.copy()
    cate_unset = self.CATE_UNSET

    from .clustering import find_clustering
    clustering = find_clustering(self, blueprint, categories)
    for cluster in np.unique(clustering):
        if cluster == 0:
            continue

        area: NDArray[np.bool_] = clustering == cluster
        if threshold is not None:
            if np.count_nonzero(area) < threshold:
                if unset:
                    ret[area] = cate_unset
                continue

        c = np.unique(ret[area])
        assert len(c) == 1
        c = int(c[0])

        s = np.unique(self.s[area])
        assert len(s) == 1
        s = int(s[0])

        x = (self.x[area] / dx).astype(int)
        y = (self.y[area] / dy).astype(int)
        y0 = int(np.min(y))
        y1 = int(np.max(y))

        for xx in np.unique(x):  # for each column
            xx = int(xx)
            if gap is None:
                for yy in range(y0, y1 + 1):
                    if not area[(yi := self._position_index[(s, xx, yy)])]:
                        ret[yi] = c
            else:
                # fill gap in y
                for yy in range(y0, y1 + 2 - gap_window):
                    yi = np.array([
                        self._position_index[(s, xx, yy + gi)]
                        for gi in range(gap_window)
                    ])
                    if np.count_nonzero(~area[yi]) <= gap:
                        ret[yi] = c

    return ret


def extend(self: BlueprintFunctions,
           blueprint: NDArray[np.int_],
           on: int,
           step: int | tuple[int, int],
           category: int = None, *,
           threshold: int | tuple[int, int] = None,
           bi: bool = True,
           overwrite: bool = False) -> NDArray[np.int_]:
    if len(blueprint) != len(self.s):
        raise ValueError()

    match threshold:
        case None | int() | (int(), int()):
            pass
        case [int(), int()]:
            threshold = tuple(threshold)
        case _:
            raise TypeError()

    if category is None:
        category = on

    match step:
        case int(step):
            step = (0, step)
        case (int(left), int(right)):
            step = left, right
        case _:
            raise TypeError()

    def _step_as_range(step: int):
        match step:
            case 0:
                return range(0, 1)
            case step if bi:
                step = abs(step)
                return range(-step, step + 1)
            case step if step > 0:
                return range(1, step + 1)
            case step if step < 0:
                return range(step, 1)
            case _:
                raise RuntimeError()

    x_steps = _step_as_range(step[0])
    y_steps = _step_as_range(step[1])

    ret = blueprint.copy()
    unset = self.CATE_UNSET

    from .clustering import find_clustering
    clustering = find_clustering(self, blueprint, [on])
    for cluster in np.unique(clustering):
        if cluster == 0:
            continue

        area: NDArray[np.bool_] = clustering == cluster
        if threshold is not None:
            size = np.count_nonzero(area)
            match threshold:
                case int(threshold):
                    if not (threshold <= size):
                        continue
                case (int(left), int(right)):
                    if not (left <= size <= right):
                        continue
                case _:
                    raise TypeError()

        extend = np.zeros_like(area, dtype=bool)
        for x in x_steps:
            for y in y_steps:
                np.logical_or(move_i(self, area, tx=x, ty=y, init=False), extend, out=extend)

        extend[area] = False
        if not overwrite:
            extend[blueprint != unset] = False

        ret[extend] = category

    return ret
