from __future__ import annotations

import functools
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, overload, Generic, Final

import numpy as np
from numpy.typing import NDArray

from chmap.probe import ProbeDesp, M, E
from chmap.util.edit.checking import use_probe
from chmap.util.utils import doc_link, SPHINX_BUILD
from chmap.views.base import ViewBase, ControllerView, V

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from chmap.util.edit.clustering import ClusteringEdges
    from chmap.util.probe_coor import ProbeCoordinate

    BLUEPRINT = NDArray[np.int_]

elif SPHINX_BUILD:
    ProbeView = 'chmap.views.probe.ProbeView'
    NpxProbeDesp = 'chmap.probe_npx.desp.NpxProbeDesp'
    AtlasBrainView = 'chmap.views.atlas.AtlasBrainView'


    class BLUEPRINT:
        """Prevent sphinx from printing BLUEPRINT as ``ndarray[int64]``"""
        pass

__all__ = ['BlueprintFunctions', 'ClusteringEdges', 'blueprint_function', 'use_probe']


def maybe_blueprint(self: BlueprintFunctions, a):
    n = len(self.s)
    return isinstance(a, np.ndarray) and a.shape == (n,) and np.issubdtype(a.dtype, np.integer)


@doc_link(BlueprintFunctions='chmap.util.util_blueprint.BlueprintFunctions')
def blueprint_function(func):
    """
    Decorate a blueprint function to make it is able to direct apply function on
    internal blueprint.

    The function should have a signature ``(blueprint, ...) -> blueprint``.

    If the first parameter blueprint is given, it works as usually. ::

        bp.func(blueprint, ...)

    If the first parameter blueprint is omitted, use {BlueprintFunctions#blueprint()} as first arguments,
    and use {BlueprintFunctions#set_blueprint()} after it returns. ::

        bp.set_blueprint(bp.func(bp.blueprint(), ...))

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
@doc_link(BlueprintScriptView='chmap.views.edit_blueprint.BlueprintScriptView')
class BlueprintFunctions(Generic[M, E]):
    """
    Provide blueprint manipulating functions. Used by {BlueprintScriptView}.

    **channelmap functions**

    .. hlist::
        :columns: 2

        * {#check_probe()}
        * {#new_channelmap()}
        * {#add_electrodes()}
        * {#del_electrodes()}
        * {#selected_electrodes()}
        * {#set_channelmap()}

    **blueprint functions**

    .. hlist::
        :columns: 2

        * {#blueprint()}
        * {#new_blueprint()}
        * {#set_blueprint()}
        * {#apply_blueprint()}
        * {#load_blueprint()}
        * {#save_blueprint()}
        * {#set()}
        * {#unset()}
        * {#__setitem__()}
        * {#__delitem__()}
        * {#merge()}
        * {#move()}
        * {#move_i()}
        * {#find_clustering()}
        * {#clustering_edges()}
        * {#edge_rastering()}
        * {#fill()}
        * {#extend()}
        * {#reduce()}

    **electrode data processing**

    .. hlist::
        :columns: 2

        * {#load_data()}
        * {#interpolate_nan()}
        * {#draw()}

    **Probe view functions**

    Call methods from {ProbeView}.

    .. hlist::
        :columns: 2

        * {#capture_electrode()}
        * {#captured_electrodes()}
        * {#set_state_for_captured()}
        * {#set_category_for_captured()}
        * {#refresh_selection()}

    **Atlas Brain image view functions**

    Call methods from {AtlasBrainView}.

    .. hlist::
        :columns: 2

        * {#atlas_add_label()}
        * {#atlas_del_label()}
        * {#atlas_clear_labels()}
        * {#atlas_set_transform()}
        * {#atlas_set_anchor()}
        * {#atlas_new_probe()}
        * {#atlas_set_anchor_on_probe()}

    **UI communicating functions**

    .. hlist::
        :columns: 2

        * {#set_status_line()}
        * {#log_message()}
        * {#use_view()}

    """

    CATE_UNSET: int
    CATE_SET: int
    CATE_FORBIDDEN: int
    CATE_LOW: int

    def __init__(self, probe: ProbeDesp[M, E], chmap: int | str | M | None):
        self.probe: Final[ProbeDesp[M, E]] = probe
        self.channelmap: Final[M | None] = chmap
        self.categories: Final[dict[str, int]] = probe.all_possible_categories()

        if chmap is not None:
            electrodes = probe.all_electrodes(chmap)
            self.s: Final[NDArray[np.int_]] = np.array([it.s for it in electrodes])
            self.x: Final[NDArray[np.int_]] = np.array([it.x for it in electrodes])
            self.y: Final[NDArray[np.int_]] = np.array([it.y for it in electrodes])
            self.dx: Final[float] = float(np.min(np.diff(np.unique(self.x))))
            self.dy: Final[float] = float(np.min(np.diff(np.unique(self.y))))
            if self.dx <= 0 or self.dy <= 0:
                raise ValueError(f'dx={self.dx}, dy={self.dy}')

            self._position_index: dict[tuple[int, int, int], int] = {
                (int(self.s[i]), int(self.x[i] / self.dx), int(self.y[i] / self.dy)): i
                for i in range(len(self.s))
            }

            self._blueprint: BLUEPRINT = np.array([it.category for it in electrodes])
        else:
            self._blueprint = None

        self._controller: ControllerView | None = None

    def __getattr__(self, item: str):
        if item.startswith('CATE_'):
            if (ret := self.categories.get(item[5:], None)) is not None:
                return ret

        raise AttributeError(item)

    # noinspection PyFinal
    def clone(self) -> Self:
        ret = object.__new__(BlueprintFunctions)
        ret.probe = self.probe
        ret.channelmap = self.channelmap
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

    # ==================== #
    # channelmap functions #
    # ==================== #

    @doc_link()
    def add_electrodes(self, e: int | list[int] | NDArray[np.int_] | NDArray[np.bool_], *, overwrite=True):
        """
         Add electrode(s) *e* into the current channelmap.

        :param e: electrode index, index list, index array or index mask.
        :param overwrite: overwrite previous selected electrode.
        :see: {ProbeDesp#add_electrode()}
        """
        electrodes = self.probe.all_electrodes(self.channelmap)
        if isinstance(e, (int, np.integer)):
            e = [electrodes[int(e)]]
        elif isinstance(e, (list, tuple)):
            e = [electrodes[int(it)] for it in e]
        else:
            e = [electrodes[int(it)] for it in np.arange((len(electrodes)))[e]]

        for t in e:
            self.probe.add_electrode(self.channelmap, t, overwrite=overwrite)

    @doc_link()
    def del_electrodes(self, e: int | list[int] | NDArray[np.int_] | NDArray[np.bool_]):
        """
        delete electrode(s) *e* from the current channelmap.

        :param e: electrode index, index list, index array or index mask.
        :see: {ProbeDesp#del_electrode()}
        """
        electrodes = self.probe.all_electrodes(self.channelmap)
        if isinstance(e, (int, np.integer)):
            e = [electrodes[int(e)]]
        elif isinstance(e, (list, tuple)):
            e = [electrodes[int(it)] for it in e]
        else:
            e = [electrodes[int(it)] for it in np.arange((len(electrodes)))[e]]

        for t in e:
            self.probe.del_electrode(self.channelmap, t)

    @doc_link()
    def selected_electrodes(self) -> NDArray[np.int_]:
        """
        The selected electrodes in the current channelmap.

        :return: electrode index array
        :see: {ProbeDesp#all_channels()}
        """
        ret = []

        pos = self._position_index

        for e in self.probe.all_channels(self.channelmap):
            p = int(e.s), int(e.x / self.dx), int(e.y / self.dy)
            if (i := pos.get(p, None)) is not None:
                ret.append(i)

        return np.unique(ret)

    def set_channelmap(self, chmap: M):
        """
        Apply the channelmap on the current channelmap.

        :param chmap:
        """
        self.probe.clear_electrode(self.channelmap)
        for t in self.probe.all_channels(chmap):
            self.probe.add_electrode(self.channelmap, t)

    # =================== #
    # blueprint functions #
    # =================== #

    def blueprint(self) -> BLUEPRINT:
        """blueprint."""
        return self._blueprint

    def new_blueprint(self) -> BLUEPRINT:
        """new empty blueprint array."""
        return np.full_like(self.s, self.CATE_UNSET)

    def set_blueprint(self, blueprint: int | BLUEPRINT | list[E]):
        """
        set blueprint.

        :param blueprint: a blueprint or a category value.
        """
        if isinstance(blueprint, int):
            self._blueprint[:] = blueprint
            return

        if isinstance(blueprint, list):
            blueprint = np.array([it.category for it in blueprint])

        if len(blueprint) != len(self.s):
            raise ValueError()

        self._blueprint = blueprint

    def apply_blueprint(self, electrodes: list[E], blueprint: BLUEPRINT = None) -> list[E]:
        """
        Apply blueprint back to electrode list.

        :param electrodes:
        :param blueprint:
        :return: *electrodes*
        """
        if blueprint is None:
            blueprint = self.blueprint()

        for e in electrodes:
            e.state = ProbeDesp.STATE_UNUSED

        for e in self.probe.all_channels(self.channelmap, electrodes):
            for t in self.probe.invalid_electrodes(self.channelmap, e, electrodes):
                t.state = ProbeDesp.STATE_FORBIDDEN
            e.state = ProbeDesp.STATE_USED

        c = {it.electrode: it for it in electrodes}
        for e, p in zip(self.probe.all_electrodes(self.channelmap), blueprint):
            if (t := c.get(e.electrode, None)) is not None:
                t.category = int(p)

        return electrodes

    def load_blueprint(self, file: str | Path) -> BLUEPRINT:
        """
        Load the blueprint from the **file**.

        :param file: file.blueprint.npy
        :return:
        """
        file = Path(file)
        if file.name.endswith('.blueprint.npy'):
            file = file.with_suffix('.blueprint.npy')

        data = np.load(file)
        self.set_blueprint(self.probe.load_blueprint(data, self.channelmap))
        return self._blueprint

    def save_blueprint(self, file: str | Path, blueprint: BLUEPRINT = None):
        """
        Save blueprint to the **file**.

        :param file:
        :param blueprint:
        :return:
        """
        if blueprint is None:
            blueprint = self._blueprint
        else:
            if len(blueprint) != len(self._blueprint):
                raise RuntimeError()

        file = Path(file)
        if file.name.endswith('.blueprint.npy'):
            file = file.with_suffix('.blueprint.npy')

        s = self.probe.all_electrodes(self.channelmap)
        for e, c in zip(s, blueprint):
            e.category = c

        np.save(file, self.probe.save_blueprint(s))

    @blueprint_function
    def set(self, blueprint: BLUEPRINT, mask: int | NDArray[np.bool_] | NDArray[np.int_], category: int) -> BLUEPRINT:
        """
        Set *category* on the blueprint with a *mask*.

        :param blueprint:
        :param mask: category value, electrode mask or electrode index
        :param category:
        :return: a (copied) blueprint
        """
        if len(blueprint) != len(self.s):
            raise ValueError()

        if isinstance(mask, (int, np.integer)):
            mask = blueprint == mask

        ret = blueprint.copy()
        ret[mask] = category
        return ret

    @blueprint_function
    def unset(self, blueprint: BLUEPRINT, mask: int | NDArray[np.bool_] | NDArray[np.int_]) -> BLUEPRINT:
        """
        unset electrodes in the *blueprint* with a *mask*.

        :param blueprint:
        :param mask: category value, electrode mask or electrode index
        :return: a (copied) blueprint
        """
        return self.set(blueprint, mask, self.CATE_UNSET)

    @doc_link()
    def __setitem__(self, mask: int | NDArray[np.bool_] | NDArray[np.int_], category: int | str):
        """
        Set a *category* to the blueprint with a *mask*.
        The new *category* only apply on unset electrodes.
        If you want to overwrite the electrode's category, please use {#set()}.

        :param mask: category value, electrode mask or electrode index
        :param category:
        :see: {#merge()}, {#set()}
        """
        blueprint = self.blueprint()
        self.set_blueprint(self.merge(blueprint, self.set(blueprint, mask, category)))

    @doc_link()
    def __delitem__(self, mask: int | NDArray[np.bool_] | NDArray[np.int_]):
        """
        unset electrodes in the *blueprint* with a *mask*.

        :param mask: category value, electrode mask or electrode index
        :see: {#unset()}
        """
        self.set_blueprint(self.set(self.blueprint(), mask, self.CATE_UNSET))

    @overload
    def merge(self, blueprint: BLUEPRINT | BlueprintFunctions) -> BLUEPRINT:
        pass

    @overload
    def merge(self, blueprint: BLUEPRINT, other: BLUEPRINT | BlueprintFunctions = None) -> BLUEPRINT:
        pass

    def merge(self, blueprint: BLUEPRINT, other: BLUEPRINT = None) -> BLUEPRINT:
        """
        Merge two blueprints. The latter blueprint won't overwrite the former result.

        ``merge(blueprint)`` works like ``merge(blueprint(), blueprint)``.

        :param blueprint: Array[category, N]
        :param other: blueprint Array[category, N]
        :return: blueprint Array[category, N]
        """
        if other is None:
            blueprint, other = self._blueprint, blueprint

        if isinstance(other, BlueprintFunctions):
            other = other.blueprint()

        n = len(self.s)
        if len(blueprint) != n or len(other) != n:
            raise ValueError()

        return np.where(blueprint != self.CATE_UNSET, blueprint, other)

    # ================== #
    # external functions #
    # ================== #

    def move(self, a: NDArray, *,
             tx: int = 0, ty: int = 0,
             mask: NDArray[np.bool_] = None,
             axis: int = 0,
             init: float = 0) -> NDArray:
        """
        Move blueprint

        :param a: Array[V, ..., N, ...], where N means all electrodes
        :param tx: x movement in um.
        :param ty: y movement in um.
        :param mask: move electrode only in mask
        :param axis: index of N
        :param init: initial value V for *a*.
        :return: moved a (copied)
        """
        from .edit.moving import move
        return move(self, a, tx=tx, ty=ty, mask=mask, axis=axis, init=init)

    def move_i(self, a: NDArray, *,
               tx: int = 0, ty: int = 0,
               mask: NDArray[np.bool_] = None,
               axis: int = 0,
               init: float = 0) -> NDArray:
        """
        Move blueprint by step.

        :param a: Array[V, ..., N, ...], where N means electrodes
        :param tx: number of dx
        :param ty: number of dy
        :param mask: move electrode only in mask
        :param axis: index of N
        :param init: initial value V for *a*.
        :return: moved a (copied)
        """
        from .edit.moving import move_i
        return move_i(self, a, tx=tx, ty=ty, mask=mask, axis=axis, init=init)

    def find_clustering(self, blueprint: BLUEPRINT,
                        categories: int | list[int] = None, *,
                        diagonal=True) -> BLUEPRINT:
        """
        find electrode clustering with the same category.

        :param blueprint: Array[category, N]
        :param categories: only for given categories.
        :param diagonal: does surrounding includes electrodes on diagonal?
        :return: Array[int, N]
        """
        from .edit.clustering import find_clustering
        return find_clustering(self, blueprint, categories, diagonal=diagonal)

    @doc_link()
    def clustering_edges(self, blueprint: BLUEPRINT,
                         categories: int | list[int] = None) -> list[ClusteringEdges]:
        """
        For each clustering block, calculate its edges.

        :param blueprint:
        :param categories:
        :return: list of edge
        """
        from .edit.clustering import clustering_edges
        return clustering_edges(self, blueprint, categories)

    def edge_rastering(self, edges: ClusteringEdges | list[ClusteringEdges], *,
                       fill=False, overwrite=False) -> BLUEPRINT:
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
    def fill(self, blueprint: BLUEPRINT,
             categories: int | list[int] = None, *,
             threshold: int = None,
             gap: int | None = 1,
             unset: bool = False) -> BLUEPRINT:
        """
        make the area occupied by categories be filled as rectangle.

        :param blueprint: Array[category, N]
        :param categories: fill area occupied by categories.
        :param threshold: only consider area which size larger than threshold.
        :param gap: fill the gap below (abs(y) <= gap). Use ``None``, fill an area as a rectangle.
        :param unset: unset small area (depends on threshold)
        :return: blueprint Array[category, N]
        """
        from .edit.moving import fill
        return fill(self, blueprint, categories, threshold=threshold, gap=gap, unset=unset)

    @blueprint_function
    def extend(self, blueprint: BLUEPRINT,
               on: int,
               step: int | tuple[int, int],
               category: int = None, *,
               threshold: int | tuple[int, int] = None,
               bi: bool = True,
               overwrite: bool = False) -> BLUEPRINT:
        """
        extend the area occupied by the category *on* with *category*.

        :param blueprint: Array[category, N]
        :param on: on which category
        :param step: expend step on y or (x, y)
        :param category: use which category value
        :param threshold: Positive value: for area which size larger than threshold (threshold<=area).
            Negative value: for area which size smaller than threshold (area<=-threshold).
            A tuple: for area which size between a range (threshold<=area<=threshold)
        :param bi: both position and negative steps direction
        :param overwrite: overwrite category value. By default, only change the unset electrode.
        :return:
        """
        from .edit.moving import extend
        return extend(self, blueprint, on, step, category, threshold=threshold, bi=bi, overwrite=overwrite)

    @blueprint_function
    def reduce(self, blueprint: BLUEPRINT,
               on: int,
               step: int | tuple[int, int], *,
               threshold: int | tuple[int, int] = None,
               bi: bool = True) -> BLUEPRINT:
        """
        reduce the area occupied by the category *on*.

        :param blueprint: Array[category, N]
        :param on: on which category
        :param step: reduce step on y or (x, y)
        :param threshold: Positive value: for area which size larger than threshold (threshold<=area).
            Negative value: for area which size smaller than threshold (area<=-threshold).
            A tuple: for area which size between a range (threshold<=area<=threshold)
        :param bi: both position and negative steps direction
        :return:
        """
        from .edit.moving import reduce
        return reduce(self, blueprint, on, step, threshold=threshold, bi=bi)

    # ==================== #
    # data process methods #
    # ==================== #

    @doc_link()
    def load_data(self, file: str | Path) -> NDArray[np.float_]:
        """
        Load a numpy array that can be parsed by {ProbeDesp#load_blueprint()}.
        The data value is read from category value for electrodes.

        Because E's category is expected as an int, this view also take it as an int by default.

        For the Neuropixels, {NpxProbeDesp} use the numpy array in this form:

           Array[int, E, (shank, col, row, state, category)]

        :param file: data file
        :return: data array.
        """
        from .edit.data import load_data
        return load_data(self, file)

    def interpolate_nan(self, a: NDArray[np.float_],
                        kernel: int | tuple[int, int] = 1,
                        f: str | Callable[[NDArray[np.float_]], float] = 'mean') -> NDArray[np.float_]:
        """
        Interpolate the NaN value in the data *a*.

        :param a:
        :param kernel: kernel size.
        :param f: interpolate method. Default use mean.
        :return:
        """
        from .edit.data import interpolate_nan
        return interpolate_nan(self, a, kernel, f)

    # ==================== #
    # controller functions #
    # ==================== #

    @doc_link(
        get_probe_desp='chmap.probe.get_probe_desp',
        BlueprintScriptView='chmap.views.edit_blueprint.BlueprintScriptView',
        RequestChannelmapTypeError='chmap.util.edit.actions.RequestChannelmapTypeError',
    )
    def check_probe(self, probe: str | type[ProbeDesp] | None = None,
                    chmap_code: int = None, *, error=True):
        """
        Check current used probe is type of *probe*.

        If it is managed by {BlueprintScriptView}, {RequestChannelmapTypeError} could be captured,
        and the request channelmap ({#new_channelmap()}) will be created when needed.

        In another way, decorator {use_probe()} can be used to annotate the probe request on a script,
        then {BlueprintScriptView} can handle the probe creating and checking before running the script.

        :param probe: request probe. It could be family name (via {get_probe_desp()}), {ProbeDesp} type or class name.
            It ``None``, checking a probe has created, and its type doesn't matter.
        :param chmap_code: request channelmap code
        :param error:
        :return: test success.
        :raise RequestChannelmapTypeError: when check failed.
        """
        from .edit.checking import check_probe, RequestChannelmapTypeError

        try:
            check_probe(self, probe, chmap_code)
        except RequestChannelmapTypeError:
            if error:
                raise
            return False
        else:
            return True

    def use_view(self, view: str | type[V]) -> V | None:
        """
        Get corresponding {ViewBase} instance if activated.

        Implement note:
            Avoiding import V at the global that might cause ``ImportError``,
            either using type name or using local import.

        :param view: view type or its type name.
        :return:
        """
        if (controller := self._controller) is not None:
            return controller.get_view(view)
        return None

    @doc_link(DataHandler='chmap.views.data.DataHandler')
    def draw(self, a: NDArray[np.float_] | None, *, view: str | type[ViewBase] = None):
        """
        Send a drawable data array *a*  to a {DataHandler}.

        :param a: Array[float, E], where E is all electrodes
        :param view: which {DataHandler}
        """
        from .edit.actions import draw
        if (controller := self._controller) is not None:
            draw(self, controller, a, view=view)

    @doc_link()
    def set_status_line(self, message: str, *, decay: float = None):
        """

        :param message: message
        :param decay: after give seconds, clear the message.
        :see: {ViewBase#set_status()}
        """
        from .edit.actions import set_status_line
        if (controller := self._controller) is not None:
            set_status_line(controller, message, decay=decay)

    @doc_link(ChannelMapEditorApp='chmap.main_bokeh.ChannelMapEditorApp')
    def log_message(self, *message: str):
        """
        Send messages to log area in GUI.

        :param message:
        :see: {ChannelMapEditorApp#log_message()}
        """
        from .edit.actions import log_message
        if (controller := self._controller) is not None:
            log_message(controller, *message)

    # ================= #
    # ProbeView related #
    # ================= #

    @doc_link()
    def new_channelmap(self, code: int | str) -> M:
        """
        Create a new channelmap with type *code*.

        :param code: channelmap type code.
        :return: new channelmap instance.
        :see: {ProbeView#reset()}
        """
        from .edit.probe import new_channelmap
        if (controller := self._controller) is not None:
            return new_channelmap(controller, code)
        else:
            raise RuntimeError()

    @doc_link()
    def capture_electrode(self, index: NDArray[np.int_] | NDArray[np.bool_],
                          state: list[int] = None):
        """
        Capture electrodes.

        :param index: index (Array[E, N]) or bool (Array[bool, E]) array.
        :param state: restrict electrodes on given states.
        :see: {ProbeView#set_captured_electrodes()}
        """
        from .edit.probe import capture_electrode
        if (controller := self._controller) is not None:
            capture_electrode(self, controller, index, state)

    @doc_link()
    def captured_electrodes(self, all=False) -> NDArray[np.int_]:
        """

        :param all: Included forbidden electrodes?
        :return: index of captured electrodes.
        :see: {ProbeView#get_captured_electrodes_index()}
        """
        from .edit.probe import captured_electrodes
        if (controller := self._controller) is not None:
            return captured_electrodes(controller, all)
        else:
            return np.array([], dtype=int)

    @doc_link()
    def set_state_for_captured(self, state: int,
                               index: NDArray[np.int_] | NDArray[np.bool_] = None):
        """
        Set state for captured electrodes.

        :param state: a state value
        :param index: index (Array[E, N]) or bool (Array[bool, E]) array.
        :see: {#capture_electrode()}
        :see: {ProbeView#set_state_for_captured()}
        """
        from .edit.probe import set_state_for_captured
        if (controller := self._controller) is not None:
            set_state_for_captured(self, controller, state, index)

    @doc_link()
    def set_category_for_captured(self, category: int,
                                  index: NDArray[np.int_] | NDArray[np.bool_] = None):
        """
        Set category for captured electrodes.

        :param category: a category value
        :param index: index (Array[E, N]) or bool (Array[bool, E]) array.
        :see: {#capture_electrode()}
        :see: {ProbeView#set_category_for_captured()}
        """
        from .edit.probe import set_category_for_captured
        if (controller := self._controller) is not None:
            set_category_for_captured(self, controller, category, index)

    @doc_link()
    def refresh_selection(self, selector: str = None):
        """
        refresh electrode selection base on current blueprint.

        :param selector:
        :see: {ProbeView#refresh_selection()}
        """
        from .edit.probe import refresh_selection
        if (controller := self._controller) is not None:
            refresh_selection(self, controller, selector)

    # ====================== #
    # AtlasBrainView related #
    # ====================== #

    @doc_link()
    def atlas_add_label(self, text: str, pos: tuple[float, float] | tuple[float, float, float] = None, *,
                        origin: str = 'bregma', color: str = 'cyan', replace=True):
        """
        Add a label on atlas brain image.

        :param text: text content.
        :param pos: text position
        :param origin: origin reference point
        :param color: label color
        :param replace: replace label which has same text content
        :see: {AtlasBrainView#add_label()}
        """
        from .edit.atlas import atlas_add_label
        if (controller := self._controller) is not None:
            atlas_add_label(controller, text, pos, origin=origin, color=color, replace=replace)

    @doc_link()
    def atlas_del_label(self, i: int | str | list[int | str]):
        """
        Remove labels from atlas brain image.

        :param i: index, or label text or list of them.
        :see: {AtlasBrainView#del_label()}
        """
        from .edit.atlas import atlas_del_label
        if (controller := self._controller) is not None:
            atlas_del_label(controller, i)

    @doc_link()
    def atlas_clear_labels(self):
        """
        Clear all labels

        :see: {AtlasBrainView#clear_labels()}
        """
        from .edit.atlas import atlas_clear_labels
        if (controller := self._controller) is not None:
            atlas_clear_labels(controller)

    # ================================= #
    # AtlasBrainView coordinate related #
    # ================================= #

    @doc_link()
    def atlas_set_transform(self, p: tuple[float, float] = None,
                            s: float | tuple[float, float] = None,
                            rt: float = None):
        """
        updating atlas image transforming.

        :param p: center position (x, y)
        :param s: scaling (sx, sy)
        :param rt: rotating degree
        :see: {BoundView#update_boundary_transform()}
        """
        from .edit.atlas import atlas_set_transform
        if (controller := self._controller) is not None:
            atlas_set_transform(controller, p, s, rt)

    @doc_link()
    def atlas_set_anchor(self, p: tuple[float, float], a: tuple[float, float] = (0, 0)):
        """
        Update atlas image boundary transform to move *a* onto *p*.

        :param p: target point on figure. figure (probe) origin as origin.
        :param a: anchor point on image, center point as origin.
        """
        from .edit.atlas import atlas_set_anchor
        if (controller := self._controller) is not None:
            atlas_set_anchor(controller, p, a)

    @doc_link()
    def atlas_new_probe(self,
                        ap: float, dv: float, ml: float,
                        shank: int = 0,
                        rx: float = 0, ry: float = 0, rz: float = 0,
                        depth: float = 0,
                        ref: str = 'bregma') -> ProbeCoordinate | None:
        """
        Create a probe coordinate instance.

        :param ap: um
        :param dv: dv
        :param ml: ml
        :param shank: shank index
        :param rx: ap rotate
        :param ry: dv rotate
        :param rz: ml rotate
        :param depth: insert depth
        :param ref: reference origin.
        :return: a probe coordinate. ``None`` if origin not set.
        """
        from .edit.atlas import atlas_new_probe
        if (controller := self._controller) is not None:
            return atlas_new_probe(controller, ap, dv, ml, shank, rx, ry, rz, depth, ref)
        return None

    @doc_link()
    def atlas_set_anchor_on_probe(self, coor: ProbeCoordinate):
        """
        Update atlas image boundary transform to anchor insertion point onto the probe.

        :param coor: probe coordinate
        """
        from .edit.atlas import atlas_set_anchor_on_probe
        if (controller := self._controller) is not None:
            atlas_set_anchor_on_probe(self, controller, coor)
