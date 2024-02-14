from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Iterator, overload, TYPE_CHECKING, NamedTuple

import numpy as np
from numpy.typing import NDArray

from chmap.util.utils import doc_link

if TYPE_CHECKING:
    from chmap.probe import ElectrodeDesp

__all__ = ['BlueprintFunctions', 'ClusteringEdges', 'blueprint_function']


def maybe_blueprint(self: BlueprintFunctions, a):
    n = len(self.s)
    return isinstance(a, np.ndarray) and a.shape == (n,) and np.issubdtype(a.dtype, np.integer)


def blueprint_function(func):
    """
    Decorate a blueprint function to make it is able to direct apply function on
    internal blueprint.

    The function should have a signature `(blueprint, ...) -> blueprint`.

    If the first parameter blueprint is given, it works as usually. ::

        func(blueprint, ...)

    If the first parameter blueprint is omitted, use `blueprint()` as first arguments,
    and use `set_blueprint()` after it returns. ::

        bp.set_blueprint(func(bp.blueprint(), ...))

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


class ClusteringEdges(NamedTuple):
    category: int
    shank: int
    edges: list[tuple[int, int, int]]  # [(x, y, corner)]
    """
    corner:
    
        3 2 1
        4 8 0
        5 6 7
        
    origin at bottom left.
        
    """

    @property
    def x(self) -> NDArray[np.int_]:
        return np.array([it[0] for it in self.edges])

    @property
    def y(self) -> NDArray[np.int_]:
        return np.array([it[1] for it in self.edges])

    def with_shank(self, s: int) -> ClusteringEdges:
        return self._replace(shank=s)

    def with_category(self, c: int) -> ClusteringEdges:
        return self._replace(category=c)

    def set_corner(self, tr: tuple[int, int],
                   tl: tuple[int, int] = None,
                   bl: tuple[int, int] = None,
                   br: tuple[int, int] = None) -> ClusteringEdges:
        if tl is None and bl is None and br is None:
            w, h = tr
            tl = -w, h
            bl = -w, -h
            br = w, -h

        offset = [None, tr, None, tl, None, bl, None, br, (0, 0)]
        edges = [
            (x + off[0], y + off[1], 8)
            for x, y, c in self.edges
            # corner at 0, 2, 4, 6 are removed
            if (off := offset[c]) is not None
        ]
        return self._replace(edges=edges)


# noinspection PyMethodMayBeStatic
@doc_link(CriteriaParser='chmap.views.edit_blueprint.CriteriaParser')
class BlueprintFunctions:
    """
    Provide blueprint manipulating functions.
    It is used by {CriteriaParser}.
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

    @classmethod
    def from_shape(cls, shape: tuple[int, int, int],
                   categories: dict[str, int],
                   xy: tuple[int, int, int, int] = (1, 0, 1, 0)) -> BlueprintFunctions:
        """

        :param shape: (shank, row, col)
        :param categories:
        :param xy:
        :return:
        """
        s, y, x = shape
        n = x * y
        yy, xx = np.mgrid[0:y, 0:x]
        xx = np.tile(xx.ravel(), s) * xy[0] + xy[1]
        yy = np.tile(yy.ravel(), s) * xy[2] + xy[3]
        ss = np.repeat(np.arange(s), n)

        return BlueprintFunctions(ss, xx, yy, categories)

    @classmethod
    def from_blueprint(cls, e: list[ElectrodeDesp],
                       categories: dict[str, int]) -> BlueprintFunctions:
        s = np.array([it.s for it in e])
        x = np.array([it.x for it in e])
        y = np.array([it.y for it in e])
        p = np.array([it.category for it in e])
        ret = BlueprintFunctions(s, x, y, categories)
        ret.set_blueprint(p)
        return ret

    def __getattr__(self, item: str):
        if item.startswith('CATE_'):
            if (ret := self._categories.get(item[5:], None)) is not None:
                return ret

        raise AttributeError(item)

    def blueprint(self) -> NDArray[np.int_]:
        return self._blueprint

    def set_blueprint(self, blueprint: NDArray[np.int_] | list[ElectrodeDesp]):
        if isinstance(blueprint, list):
            blueprint = np.array([it.category for it in blueprint])

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

    def surrounding(self, i: int | tuple[int, int, int], *, diagonal=True) -> Iterator[int]:
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
            p = self._surrounding((s, x, y), c)
            if (i := pos.get(p, None)) is not None:
                yield i

    def _surrounding(self, i: int | tuple[int, int, int], p: int) -> tuple[int, int, int]:
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

    @blueprint_function
    def set(self, blueprint: NDArray[np.int_], mask: int | NDArray[np.bool_], category: int | str) -> NDArray[np.int_]:
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

        if isinstance(mask, (int, np.integer)):
            mask = blueprint == mask

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
                        categories: list[int | str] = None,
                        diagonal=True) -> NDArray[np.int_]:
        """
        find electrode clustering with the same category.

        :param blueprint: Array[category, N]
        :param categories: only for given categories.
        :param diagonal: does surrounding includes electrodes on diagonal?
        :return: Array[int, N]
        """
        if len(blueprint) != len(self.s):
            raise ValueError()

        if categories is not None:
            categories = self.as_category(categories)

        ret: NDArray[np.int_] = np.arange(len(blueprint)) + 1

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

        for i in range(len(blueprint)):
            if ret[i] > 0:
                for j in self.surrounding(i, diagonal=diagonal):
                    if blueprint[i] == blueprint[j]:
                        union(i, j)

        return ret

    def clustering_edges(self, blueprint: NDArray[np.int_],
                         categories: list[int | str] = None) -> list[ClusteringEdges]:
        """
        For each clustering block, calculate its edges.

        :param blueprint:
        :param categories:
        :return: list of ClusteringEdges
        """
        dx = self.dx
        dy = self.dy

        clustering = self.find_clustering(blueprint, categories, diagonal=False)
        ret = []

        for cluster in np.unique(clustering):
            if cluster == 0:
                continue

            area: NDArray[np.bool_] = clustering == cluster

            c = np.unique(blueprint[area])
            assert len(c) == 1
            c = int(c[0])

            s = np.unique(self.s[area])
            assert len(s) == 1
            s = int(s[0])

            x = self.x[area]
            y = self.y[area]

            if np.count_nonzero(area) == 1:
                ret.append(ClusteringEdges(c, s, [(x, y, 1), (x, y, 3), (x, y, 5), (x, y, 7)]))
            else:
                x0 = int(np.min(x))
                y0 = int(np.min(y[x == x0]))

                i = self._position_index[(s, int(x0 / dx), int(y0 / dy))]
                ret.append(ClusteringEdges(c, s, self._cluster_edge(area, i)))

        return ret

    def _cluster_edge(self, area: NDArray[np.bool_], i: int) -> list[tuple[int, int, int]]:
        """

        :param area:
        :param i: start index
        :return: list of (x, y, corner)
        """
        if not area[i]:
            raise ValueError(f'no cluster at index {i}')

        pos = self._position_index

        # 3 2 1
        # 4 e 0
        # 5 6 7

        actions = {
            # direction (i -> j):
            0: {  # rightward
                # next direction (j -> k): corner
                # None: (corners), action

                # * * *
                # i j *
                #   k *
                6: 5,

                # * * *
                # i j k
                # ?
                0: 6,

                # * k
                # i j
                #
                2: 7,

                # ?
                # i j
                #
                None: ((7, 1, 3), 2)
            },
            6: {  # downward
                #   i *
                # k j *
                # * * *
                4: 3,

                # ? i *
                #   j *
                #   k *
                6: 4,

                # ? i *
                #   j k
                #
                0: 5,

                # ? i ?
                #   j
                #
                None: ((5, 7, 1), 0)
            },
            4: {  # leftward
                # ? k
                # * j i
                # * * *
                2: 1,

                #
                # k j i
                # * * *
                4: 2,

                #
                #   j i
                # ? k *
                6: 3,

                #
                #   j i
                #     ?
                None: ((3, 5, 7), 6)
            },
            2: {  # upward
                # * * *
                # * j k
                # * i
                0: 7,

                # * k
                # * j
                # * i
                2: 0,

                #
                # k j
                # * i
                4: 1,

                #
                #   j
                # ? i
                None: ((1, 3, 5), 4)
            }
        }

        x = self.x[i]
        y = self.y[i]
        ret = [(x, y, 5)]
        # * ?
        # i * ?
        #   ? ?
        j = i
        d = 0  # right
        while not (i == j and d == 6):
            if not area[j]:
                raise ValueError(f'no cluster at index j={j}')

            # print(debug_print_local(self, area.astype(int), j, size=2))
            x = self.x[j]
            y = self.y[j]
            for action, corner in actions[d].items():
                if action is not None:
                    if (k := pos.get(self._surrounding(j, action), None)) is not None and area[k]:
                        ret.append((x, y, corner))
                        j = k
                        d = action
                        break
                    else:
                        continue
                elif action is None:
                    corner, action = corner
                    for _corner in corner:
                        ret.append((x, y, _corner))
                    d = action
                    break
            else:
                raise RuntimeError('un-reachable')

        return ret

    def edge_rastering(self, edges: ClusteringEdges | list[ClusteringEdges], fill=False, overwrite=False) -> NDArray[np.int_]:
        """
        For given edges, put them on the blueprint.

        :param edges:
        :param fill: fill the area.
        :param overwrite: latter result overwrite previous results
        :return: blueprint
        """
        match edges:
            case (ClusteringEdges() as edge) | [ClusteringEdges() as edge]:
                return self._edge_rastering(edge, fill=fill)

        unset = self.CATE_UNSET
        ret = np.full_like(self.s, unset)
        for edge in edges:
            res = self._edge_rastering(edge, fill=fill)
            if overwrite:
                ret = np.where(res == unset, ret, res)
            else:
                ret = np.where(ret == unset, res, ret)
        return ret

    def _edge_rastering(self, edge: ClusteringEdges, fill=False) -> NDArray[np.int_]:
        dx = self.dx
        dy = self.dy
        pos = self._position_index

        c = edge.category
        s = edge.shank
        edge = edge.set_corner((0, 0))
        edge = [*edge.edges, edge.edges[0]]  # as closed polygon

        unset = self.CATE_UNSET
        ret = np.full_like(self.s, unset)

        for i in range(len(edge) - 1):
            px, py, _ = edge[i]
            qx, qy, _ = edge[i + 1]
            if px == qx:
                x = int(px / dx)
                py, qy = min(py, qy), max(py, qy)
                for y in range(py, qy + 1):
                    y = int(y / dy)
                    if (p := pos.get((s, x, y), None)) is not None:
                        ret[p] = c
            elif py == qy and not fill:
                y = int(py / dy)
                px, qx = min(px, qx), max(px, qx)
                for x in range(px, qx + 1):
                    x = int(x / dx)
                    if (p := pos.get((s, x, y), None)) is not None:
                        ret[p] = c

        if not fill:
            return ret

        n_x = len(set([
            x for _s, x, y in self._position_index
            if _s == s
        ]))

        ret = ret.reshape((-1, n_x))  # (Y, X)
        interior = np.cumsum((ret != unset).astype(int), axis=1) % 2 == 1
        ret[interior] = c

        return ret.ravel()

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


def debug_print_local(bp: BlueprintFunctions, data: NDArray, i: int, size: int = 1) -> str:
    s = int(bp.s[i])
    x = int(bp.x[i] / bp.dx)
    y = int(bp.y[i] / bp.dy)

    ret = []
    for dy in range(-size, size + 1):
        ret.append((row := []))
        for dx in range(-size, size + 1):
            j = bp._position_index.get((s, x + dx, y + dy), None)
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
