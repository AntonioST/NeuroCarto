from __future__ import annotations

import functools
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from chmap.probe import ProbeDesp, M, E
from chmap.util.utils import doc_link
from .edit import validation

if TYPE_CHECKING:
    from chmap.views.base import ViewBase, ControllerView
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
class BlueprintFunctions:
    """
    Provide blueprint manipulating functions.
    """

    CATE_UNSET: int = ProbeDesp.CATE_UNSET
    CATE_SET: int = ProbeDesp.CATE_SET
    CATE_FORBIDDEN: int = ProbeDesp.CATE_FORBIDDEN
    CATE_LOW: int = ProbeDesp.CATE_LOW

    def __init__(self, probe: ProbeDesp[M, E], chmap: M):
        self.probe = probe
        self.chmap = chmap
        self.categories = probe.all_possible_categories()

        electrodes = probe.all_electrodes(chmap)
        self.s = np.array([it.s for it in electrodes])
        self.x = np.array([it.x for it in electrodes])
        self.y = np.array([it.y for it in electrodes])
        self.dx = np.min(np.diff(np.unique(self.x)))
        self.dy = np.min(np.diff(np.unique(self.y)))
        if self.dx <= 0 or self.dy <= 0:
            raise ValueError(f'dx={self.dx}, dy={self.dy}')

        self._position_index = {
            (int(self.s[i]), int(self.x[i] / self.dx), int(self.y[i] / self.dy)): i
            for i in range(len(self.s))
        }

        self._blueprint: NDArray[np.int_] = np.array([it.category for it in electrodes])
        self._controller: ControllerView = None

    def clone(self) -> BlueprintFunctions:
        ret = object.__new__(BlueprintFunctions)
        ret.probe = self.probe
        ret.chmap = ret.probe.new_channelmap(self.chmap)
        ret.categories = self.categories

        ret.s = self.s
        ret.x = self.x
        ret.y = self.y
        ret.dx = self.dx
        ret.dy = self.dy
        ret._position_index = self._position_index
        ret._blueprint = self._blueprint.copy()
        ret._controller = self._controller
        return ret

    def check_probe(self, probe: str | type[ProbeDesp], *, error=True) -> bool:
        if isinstance(probe, str):
            test = type(self.probe).__name__ == probe
        elif isinstance(probe, type):
            test = isinstance(self.probe, probe)
        else:
            raise TypeError()

        if not test:
            if error:
                raise RuntimeError()
        return test

    def __getattr__(self, item: str):
        if item.startswith('CATE_'):
            if (ret := self.categories.get(item, None)) is not None:
                return ret

        raise AttributeError(item)

    def blueprint(self) -> NDArray[np.int_]:
        return self._blueprint

    def set_blueprint(self, blueprint: NDArray[np.int_] | list[E]):
        if isinstance(blueprint, list):
            blueprint = np.array([it.category for it in blueprint])

        if len(blueprint) != len(self.s):
            raise ValueError()

        self._blueprint = blueprint

    def load_blueprint(self, file: str | Path) -> NDArray[np.int_]:
        """

        :param file: file.blueprint.npy
        :return:
        """
        file = Path(file)
        if file.name.endswith('.blueprint.npy'):
            file = file.with_suffix('.blueprint.npy')

        data = np.load(file)
        self.set_blueprint(self.probe.load_blueprint(data, self.chmap))
        return self._blueprint

    def save_blueprint(self, file: str | Path, blueprint: NDArray[np.int_] = None):
        if blueprint is None:
            blueprint = self._blueprint
        else:
            if len(blueprint) != len(self._blueprint):
                raise RuntimeError()

        file = Path(file)
        if file.name.endswith('.blueprint.npy'):
            file = file.with_suffix('.blueprint.npy')

        s = self.probe.all_electrodes(self.chmap)
        for e, c in zip(s, blueprint):
            e.category = c

        np.save(file, self.probe.save_blueprint(s))

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

    # ================== #
    # external functions #
    # ================== #

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
                        categories: int | list[int] = None, *,
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
                         categories: int | list[int] = None) -> list[ClusteringEdges]:
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
             categories: int | list[int] = None, *,
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
               on: int,
               step: int | tuple[int, int],
               category: int = None, *,
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

    # ====================== #
    # data input validations #
    # ====================== #

    arg = validation

    # ==================== #
    # data process methods #
    # ==================== #

    def log_message(self, *message: str):
        if (controller := self._controller) is None:
            return

        from chmap.views.base import ViewBase
        if isinstance(controller, ViewBase):
            controller.log_message(*message)

    @doc_link()
    def load_data(self, file: str | Path) -> NDArray[np.float_]:
        """
        Load a numpy array that can be parsed by {ProbeDesp#load_blueprint()}.
        The data value is read from category value for electrodes.

        Because E's category is expected as an int, this view also take it as an int by default.

        For the Neuropixels, `NpxProbeDesp` use the numpy array in this form:

           Array[int, E, (shank, col, row, state, category)]

        :param file:
        :return:
        """
        from .edit.data import load_data
        return load_data(self, file)

    def draw(self, a: NDArray[np.float_] | None, *, view: str | type[ViewBase] = None):
        if (controller := self._controller) is None:
            return

        from chmap.views.data import DataHandler
        if isinstance(controller, DataHandler):
            controller.on_data_update(self.probe, self.probe.all_electrodes(self.chmap), a)
        elif isinstance(view_target := controller.get_view(view), DataHandler):
            view_target.on_data_update(self.probe, self.probe.all_electrodes(self.chmap), a)

    def interpolate_nan(self, a: NDArray[np.float_],
                        kernel: int | tuple[int, int] = 1,
                        f: str | Callable[[NDArray[np.float_]], float] = 'mean') -> NDArray[np.float_]:
        from .edit.data import interpolate_nan
        return interpolate_nan(self, a, kernel, f)
