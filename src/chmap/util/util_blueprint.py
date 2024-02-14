from __future__ import annotations

import functools
from collections.abc import Callable
from typing import overload, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from chmap.util.utils import doc_link

if TYPE_CHECKING:
    from chmap.probe import ElectrodeDesp
    from chmap.util.edit.clustering import ClusteringEdges

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

    def __setitem__(self, mask: int | NDArray[np.bool_], category: int | str):
        self.set_blueprint(self.set(self.blueprint(), mask, category))

    def __delitem__(self, mask: int | NDArray[np.bool_]):
        self.set_blueprint(self.set(self.blueprint(), mask, self.CATE_UNSET))

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
        from .edit.moving import move
        return move(self, a, tx=tx, ty=ty, shanks=shanks, axis=axis, init=init)

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
        from .edit.moving import move_i
        return move_i(self, a, tx=tx, ty=ty, shanks=shanks, axis=axis, init=init)

    def find_clustering(self, blueprint: NDArray[np.int_],
                        categories: list[int | str] = None, *,
                        diagonal=True) -> NDArray[np.int_]:
        """
        find electrode clustering with the same category.

        :param blueprint: Array[category, N]
        :param categories: only for given categories.
        :param diagonal: does surrounding includes electrodes on diagonal?
        :return: Array[int, N]
        """
        from .edit.clustering import find_clustering
        return find_clustering(self, blueprint, categories, diagonal=diagonal)

    def clustering_edges(self, blueprint: NDArray[np.int_],
                         categories: list[int | str] = None) -> list[ClusteringEdges]:
        """
        For each clustering block, calculate its edges.

        :param blueprint:
        :param categories:
        :return: list of ClusteringEdges
        """
        from .edit.clustering import clustering_edges
        return clustering_edges(self, blueprint, categories)

    def edge_rastering(self, edges: ClusteringEdges | list[ClusteringEdges], *,
                       fill=False, overwrite=False) -> NDArray[np.int_]:
        """
        For given edges, put them on the blueprint.

        :param edges:
        :param fill: fill the area.
        :param overwrite: latter result overwrite previous results
        :return: blueprint
        """
        from .edit.clustering import edge_rastering
        return edge_rastering(self, edges, fill=fill, overwrite=overwrite)

    @blueprint_function
    def fill(self, blueprint: NDArray[np.int_],
             categories: int | str | list[int | str] = None, *,
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
        from .edit.moving import fill
        return fill(self, blueprint, categories, threshold=threshold, gap=gap, unset=unset)

    @blueprint_function
    def extend(self, blueprint: NDArray[np.int_],
               on: int | str,
               step: int | tuple[int, int],
               category: int | str = None, *,
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
        from .edit.moving import extend
        return extend(self, blueprint, on, step, category, threshold=threshold, bi=bi, overwrite=overwrite)

    def interpolate_nan(self, a: NDArray[np.float_],
                        kernel: int | tuple[int, int] = 1,
                        f: str | Callable[[NDArray[np.float_]], float] = 'mean') -> NDArray[np.float_]:
        from .edit.data import interpolate_nan
        return interpolate_nan(self, a, kernel, f)
