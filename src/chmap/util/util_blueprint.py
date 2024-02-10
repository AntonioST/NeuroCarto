from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Iterator, overload

import numpy as np
from numpy.typing import NDArray

__all__ = ['BlueprintFunctions']


def maybe_blueprint(self: BlueprintFunctions, a):
    n = len(self.s)
    return isinstance(a, np.ndarray) and a.shape == (n,) and np.issubdtype(a.dtype, np.integer)


def blueprint_function(func):
    """
    Decorate a blueprint function to make it is able to direct apply function on
    internal blueprint.

    The function should have a signature `(blueprint, ...) -> blueprint`.

    If the first parameter blueprint is given, it works as usually.
    If the first parameter blueprint is omitted, use `blueprint()` as first arguments,
    and use `set_blueprint()` when it returns.

    :param func:
    :return:
    """

    @functools.wraps(func)
    def _blueprint_function(self: BlueprintFunctions, *args, **kwargs):
        if len(args) and maybe_blueprint(self, args[0]):
            return func(self, *args, **kwargs)
        else:
            blueprint = self.blueprint()
            ret = func(self, blueprint, *args, **kwargs)
            if maybe_blueprint(self, ret):
                self.set_blueprint(ret)
            return ret

    return _blueprint_function


# noinspection PyMethodMayBeStatic
class BlueprintFunctions:
    """
    Provide blueprint manipulating functions.
    It is used by `chmap.views.edit_blueprint.CriteriaParser`.
    """

    CATE_UNSET: int
    CATE_SET: int
    CATE_FORBIDDEN: int
    CATE_LOW: int

    def __init__(self, s: NDArray[np.int_], x: NDArray[np.int_], y: NDArray[np.int_],
                 categories: dict[str, int]):
        if s.ndim != 1 or x.ndim != 1 or y.ndim != 1:
            raise ValueError()
        if (n := len(s)) != len(x) or n != len(y) or n == 0:
            raise ValueError()

        self.s = s
        self.x = x
        self.y = y
        self.dx = np.min(np.diff(np.unique(x)))
        self.dy = np.min(np.diff(np.unique(y)))
        if self.dx <= 0 or self.dy <= 0:
            raise ValueError(f'dx={self.dx}, dy={self.dy}')

        self._position_index = {
            (int(s[i]), int(x[i] / self.dx), int(y[i] / self.dy)): i
            for i in range(len(s))
        }

        self._categories = categories

        self._blueprint: NDArray[np.int_] = None

    def __getattr__(self, item: str):
        if item.startswith('CATE_'):
            if (ret := self._categories.get(item[5:], None)) is not None:
                return ret

        raise AttributeError(item)

    def blueprint(self) -> NDArray[np.int_]:
        return self._blueprint

    def set_blueprint(self, blueprint: NDArray[np.int_]):
        if len(blueprint) != len(self.s):
            raise ValueError()

        self._blueprint = blueprint

    @overload
    def as_category(self, category: int | str) -> int:
        pass

    @overload
    def as_category(self, category: list[int | str]) -> list[int]:
        pass

    def as_category(self, category):
        match category:
            case int(category):
                return category
            case str(category):
                return getattr(self, f'CATE_{category.upper()}')
            case list():
                return list(map(self.as_category, category))
            case _:
                raise TypeError()

    def move(self, a: NDArray, *,
             tx: int = 0, ty: int = 0,
             shanks: list[int] = None,
             axis: int = 0,
             init: float = 0) -> NDArray:
        """
        Move blueprint

        :param a: Array[V, ..., N, ...], where N means all electrodes
        :param tx: x movement in um.
        :param ty: y movement in um.
        :param shanks: move electrode only on given shanks
        :param axis: index off N
        :param init: initial value
        :return: moved a (copied)
        """
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

    def move_i(self, a: NDArray, *,
               tx: int = 0, ty: int = 0,
               shanks: list[int] = None,
               axis: int = 0,
               init: float = 0) -> NDArray:
        """
        Move blueprint by step.

        :param a: Array[V, ..., N, ...], where N means electrodes
        :param tx: number of dx
        :param ty: number of dy
        :param shanks: move electrode only on given shanks
        :param axis: index off N
        :param init: initial value
        :return: moved a (copied)
        """
        if tx == 0 and ty == 0:
            return a
        return self.move(a, tx=tx * self.dx, ty=ty * self.dy, shanks=shanks, axis=axis, init=init)

    @blueprint_function
    def set(self, blueprint: NDArray[np.int_], mask: NDArray[np.bool_], category: int | str) -> NDArray[np.int_]:
        """
        set *category* on `blueprint[mask]`.

        :param blueprint:
        :param mask:
        :param category:
        :return:
        """
        if len(blueprint) != len(self.s):
            raise ValueError()

        if isinstance(category, str):
            category = self._categories[category]

        ret = blueprint.copy()
        ret[mask] = category
        return ret

    @blueprint_function
    def unset(self, blueprint: NDArray[np.int_], mask: NDArray[np.bool_]) -> NDArray[np.int_]:
        """
        unset `blueprint[mask]`.

        :param blueprint:
        :param mask:
        :return:
        """
        return self.set(blueprint, mask, self.CATE_UNSET)

    def merge(self, blueprint: NDArray[np.int_], other: NDArray[np.int_] = None) -> NDArray[np.int_]:
        """
        merge blueprint. The latter result overwrite former result.

        `merge(blueprint)` works like `merge(blueprint(), blueprint)`.

        :param blueprint: Array[category, N]
        :param other: blueprint Array[category, N]
        :return: blueprint Array[category, N]
        """
        if other is None:
            if self._blueprint is None:
                return blueprint

            other = blueprint
            blueprint = self._blueprint

        n = len(self.s)
        if len(blueprint) != n or len(other) != n:
            raise ValueError()

        return np.where(other == self.CATE_SET, blueprint, other)

    def interpolate_nan(self, a: NDArray[np.float_],
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
                r.append(self.move_i(a, tx=tx, ty=ty, init=np.nan))

        r = f(r, axis=0)

        ret = a.copy()
        ret[m] = r[m]
        return ret

    def find_clustering(self, blueprint: NDArray[np.int_],
                        categories: list[int | str] = None) -> NDArray[np.int_]:
        """
        find electrode clustering with the same category.

        :param blueprint: Array[category, N]
        :param categories: only for given categories.
        :return: Array[int, N]
        """
        s = self.s
        x = self.x
        y = self.y
        dx = self.dx
        dy = self.dy

        if len(blueprint) != len(s):
            raise ValueError()

        if categories is not None:
            categories = self.as_category(categories)

        pos = self._position_index

        ret: NDArray[np.int_] = np.arange(len(s)) + 1

        unset = self.CATE_UNSET
        ret[blueprint == unset] = 0
        if categories is not None:
            for category in np.unique(blueprint):
                if int(category) not in categories:
                    ret[blueprint == category] = 0

        def union(i: int, j: int):
            if i == j:
                return

            a: int = ret[i]
            b: int = ret[j]
            c = min(a, b)

            if a != c:
                ret[ret == a] = c
            if b != c:
                ret[ret == b] = c

        def surr(i) -> Iterator[tuple[int, int, int]]:
            ss = int(s[i])
            xx = int(x[i] / dx)
            yy = int(y[i] / dy)
            # 3 2 1
            # 4 e 0
            # 5 6 7
            yield ss, xx + 1, yy
            yield ss, xx + 1, yy + 1
            yield ss, xx, yy + 1
            yield ss, xx - 1, yy + 1
            yield ss, xx - 1, yy
            yield ss, xx - 1, yy - 1
            yield ss, xx, yy - 1
            yield ss, xx + 1, yy - 1

        for i in range(len(s)):
            if ret[i] > 0:
                for p in surr(i):
                    if (j := pos.get(p, None)) is not None and blueprint[i] == blueprint[j]:
                        union(i, j)

        return ret

    @blueprint_function
    def fill(self, blueprint: NDArray[np.int_],
             categories: int | str | list[int | str] = None,
             threshold: int = None,
             gap: int | None = 1,
             unset: bool = False) -> NDArray[np.int_]:
        """
        make the area occupied by categories be filled as rectangle.

        :param blueprint: Array[category, N]
        :param categories: fill area occupied by categories.
        :param threshold: only consider area which size larger than threshold.
        :param gap: fill the gap below (|y| <= gap). Use None, fill() area as a rectangle.
        :param unset: unset small area (depends on threshold)
        :return: blueprint Array[category, N]
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
            categories = list(set(self._categories.values()))
        elif isinstance(categories, (int, str)):
            categories = [categories]

        dx = self.dx
        dy = self.dy

        ret = blueprint.copy()
        cate_unset = self.CATE_UNSET

        clustering = self.find_clustering(blueprint, categories)
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

    @blueprint_function
    def extend(self, blueprint: NDArray[np.int_],
               on: int | str,
               step: int | tuple[int, int],
               category: int | str = None,
               threshold: int | tuple[int, int] = None,
               bi: bool = True,
               overwrite: bool = False):
        """
        extend the area occupied by category *on* with *category*.

        :param blueprint: Array[category, N]
        :param on: on which category
        :param step: expend step on y or (x, y)
        :param category: use which category value
        :param threshold: for area which size larger than threshold (threshold<=|area|)
            or in range (threshold<=|area|<=threshold)
        :param bi: both position and negative steps direction
        :param overwrite: overwrite category value. By default, only change the unset electrode.
        :return:
        """
        if len(blueprint) != len(self.s):
            raise ValueError()

        on = self.as_category(on)

        match threshold:
            case None | int() | (int(), int()):
                pass
            case [int(), int()]:
                threshold = tuple(threshold)
            case _:
                raise TypeError()

        if category is None:
            category = on
        else:
            category = self.as_category(category)

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

        clustering = self.find_clustering(blueprint, [on])
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
                    np.logical_or(self.move_i(area, tx=x, ty=y, init=False), extend, out=extend)

            extend[area] = False
            if not overwrite:
                extend[blueprint != unset] = False

            ret[extend] = category

        return ret
