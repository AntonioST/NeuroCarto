import textwrap

import numpy as np
from numpy.typing import NDArray

from neurocarto.util.util_blueprint import BlueprintFunctions
from neurocarto.util.utils import doc_link

__all__ = [
    'move', 'move_i',
    'fill', 'extend', 'reduce'
]


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.move.__doc__))
def move(self: BlueprintFunctions, a: NDArray, *,
         tx: int = 0, ty: int = 0,
         mask: NDArray[np.bool_] = None,
         axis: int = 0,
         init: float = 0) -> NDArray:
    """
    {DOC}
    :see: {BlueprintFunctions#move()}
    """
    s = self.s
    x = self.x
    y = self.y
    dx = self.dx
    dy = self.dy

    if a.shape[axis] != len(s):
        raise RuntimeError()

    if mask is not None and len(mask) != len(s):
        raise RuntimeError()

    if abs(tx) < dx and abs(ty) < dy:
        return a

    pos = self._position_index

    ii = np.arange(len(s))
    jj = np.arange(len(s))
    rm = []
    for i in range(len(s)):
        if mask is None or mask[i]:
            p = int(s[i]), int((x[i] + tx) / dx), int((y[i] + ty) / dy)
            if (j := pos.get(p, None)) is None:
                rm.append(i)
            else:
                jj[i] = j
        else:
            rm.append(i)

    if len(rm):
        ii = np.delete(ii, rm)
        jj = np.delete(jj, rm)

    if a.ndim > 1:
        _index: list[slice | NDArray[np.int_]] = [slice(None)] * a.ndim
        _index[axis] = ii
        ii = tuple(_index)
        _index[axis] = jj
        jj = tuple(_index)

    ret = np.full_like(a, init)
    ret[jj] = a[ii]

    return ret


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.move_i.__doc__))
def move_i(self: BlueprintFunctions, a: NDArray, *,
           tx: int = 0, ty: int = 0,
           mask: NDArray[np.bool_] = None,
           axis: int = 0,
           init: float = 0) -> NDArray:
    """
    {DOC}
    :see: {BlueprintFunctions#move_i()}
    """
    if tx == 0 and ty == 0:
        return a

    s = self.s
    x = (self.x / self.dx).astype(int)
    y = (self.y / self.dy).astype(int)

    if a.shape[axis] != len(s):
        raise RuntimeError()

    if mask is not None and len(mask) != len(s):
        raise RuntimeError()

    pos = self._position_index

    ii = np.arange(len(s))
    jj = np.arange(len(s))
    rm = []
    for i in ii:
        if mask is None or mask[i]:
            p = int(s[i]), int(x[i] + tx), int(y[i] + ty)
            if (j := pos.get(p, None)) is None:
                rm.append(i)
            else:
                jj[i] = j
        else:
            rm.append(i)

    if len(rm):
        ii = np.delete(ii, rm)
        jj = np.delete(jj, rm)

    if a.ndim > 1:
        _index: list[slice | NDArray[np.int_]] = [slice(None)] * a.ndim
        _index[axis] = ii
        ii = tuple(_index)
        _index[axis] = jj
        jj = tuple(_index)

    ret = np.full_like(a, init)
    ret[jj] = a[ii]

    return ret


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.fill.__doc__))
def fill(self: BlueprintFunctions,
         blueprint: NDArray[np.int_],
         categories: int | list[int] = None, *,
         threshold: int = None,
         gap: int | None = 1,
         unset: bool = False) -> NDArray[np.int_]:
    """
    {DOC}
    :see: {BlueprintFunctions#fill()}
    """
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
                assert gap_window is not None
                # fill gap in y
                for yy in range(y0, y1 + 2 - gap_window):
                    yi = np.array([
                        self._position_index[(s, xx, yy + gi)]
                        for gi in range(gap_window)
                    ])
                    if np.count_nonzero(~area[yi]) <= gap:
                        ret[yi] = c

    return ret


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.extend.__doc__))
def extend(self: BlueprintFunctions,
           blueprint: NDArray[np.int_],
           category: int,
           step: int | tuple[int, int],
           value: int = None, *,
           threshold: int | tuple[int, int] = None,
           bi: bool = True,
           overwrite: bool = False) -> NDArray[np.int_]:
    """
    {DOC}
    :see: {BlueprintFunctions#extend()}
    """
    if len(blueprint) != len(self.s):
        raise ValueError()

    match threshold:
        case None | int() | (int(), int()):
            pass
        case [int(), int()]:
            threshold = tuple(threshold)
        case _:
            raise TypeError()

    if value is None:
        value = category

    match step:
        case int(step):
            step = (0, step)
        case (int(left), int(right)):
            step = left, right
        case _:
            raise TypeError()

    x_steps = _step_as_range(step[0], bi)
    y_steps = _step_as_range(step[1], bi)

    ret = blueprint.copy()
    unset = self.CATE_UNSET

    from .clustering import find_clustering
    clustering = find_clustering(self, blueprint, [category])
    for cluster in np.unique(clustering):
        if cluster == 0:
            continue

        area: NDArray[np.bool_] = clustering == cluster
        if threshold is not None and not _check_area_size(np.count_nonzero(area), threshold):
            continue

        extend = np.zeros_like(area, dtype=bool)
        for x in x_steps:
            for y in y_steps:
                np.logical_or(move_i(self, area, tx=x, ty=y, mask=area, init=False), extend, out=extend)

        extend[area] = False
        if not overwrite:
            extend[blueprint != unset] = False

        ret[extend] = value

    return ret


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.reduce.__doc__))
def reduce(self: BlueprintFunctions,
           blueprint: NDArray[np.int_],
           category: int,
           step: int | tuple[int, int], *,
           threshold: int | tuple[int, int] = None,
           bi: bool = True) -> NDArray[np.int_]:
    """
    {DOC}
    :see: {BlueprintFunctions#reduce()}
    """
    if len(blueprint) != len(self.s):
        raise ValueError()

    match threshold:
        case None | int() | (int(), int()):
            pass
        case [int(), int()]:
            threshold = tuple(threshold)
        case _:
            raise TypeError()

    match step:
        case int(step):
            step = (0, step)
        case (int(left), int(right)):
            step = left, right
        case _:
            raise TypeError()

    x_steps = _step_as_range(step[0], bi)
    y_steps = _step_as_range(step[1], bi)

    ret = blueprint.copy()
    unset = self.CATE_UNSET

    from .clustering import find_clustering
    clustering = find_clustering(self, blueprint, [category])
    for cluster in np.unique(clustering):
        if cluster == 0:
            continue

        area: NDArray[np.bool_] = clustering == cluster
        if threshold is not None and not _check_area_size(np.count_nonzero(area), threshold):
            continue

        inner = area.copy()
        for x in x_steps:
            for y in y_steps:
                np.logical_and(move_i(self, area, tx=x, ty=y, mask=area, init=False), inner, out=inner)

        remove = np.logical_and(area, ~inner)

        ret[remove] = unset

    return ret


def _step_as_range(step: int, bi: bool) -> range:
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
    raise RuntimeError()


def _check_area_size(area: int, threshold: int | tuple[int, int]) -> bool:
    match threshold:
        case int(threshold) if threshold >= 0:
            return threshold <= area
        case int(threshold) if threshold < 0:
            return area <= -threshold
        case (int(left), int(right)):
            return left <= area <= right
    raise TypeError()
