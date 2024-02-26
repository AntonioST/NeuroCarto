from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from chmap.probe import ProbeDesp
from chmap.util.util_blueprint import BlueprintFunctions
from chmap.util.utils import SPHINX_BUILD, doc_link
from chmap.views.base import ViewBase, ControllerView
from chmap.views.data import DataHandler

if TYPE_CHECKING:
    from chmap.views.atlas import AtlasBrainView
elif SPHINX_BUILD:
    ViewBase = 'chmap.views.base.ViewBase'
    ProbeView = 'chmap.views.probe.ProbeView'
    AtlasBrainView = 'chmap.views.atlas.AtlasBrainView'

__all__ = [
    'new_channelmap',
    'log_message',
    'set_status_line',
    'draw',
    'capture_electrode',
    'captured_electrodes',
    'set_state_for_captured',
    'set_category_for_captured',
    'refresh_selection',
    'atlas_add_label',
    'atlas_del_label',
    'atlas_clear_labels'
]


@doc_link()
def new_channelmap(controller: ControllerView, code: int | str) -> Any:
    """{ProbeView#reset()}"""
    app = controller.get_app()
    app.on_new(code)
    return app.probe_view.channelmap


@doc_link()
def log_message(controller: ControllerView, *message: str):
    """{ViewBase#log_message()}"""
    if isinstance(controller, ViewBase):
        controller.log_message(*message)


@doc_link()
def set_status_line(controller: ControllerView, message: str, *, decay: float = None):
    """{ViewBase#set_status()}"""
    if isinstance(controller, ViewBase):
        controller.set_status(message, decay=decay)


@doc_link()
def draw(self: BlueprintFunctions, controller: ControllerView,
         a: NDArray[np.float_] | None, *,
         view: str | type[ViewBase] = None):
    """{DataHandler#on_data_update()}"""
    if isinstance(controller, DataHandler):
        controller.on_data_update(self.probe, self.probe.all_electrodes(self.channelmap), a)
    elif isinstance(view_target := controller.get_view(view), DataHandler):
        view_target.on_data_update(self.probe, self.probe.all_electrodes(self.channelmap), a)


@doc_link()
def capture_electrode(self: BlueprintFunctions, controller: ControllerView,
                      index: NDArray[np.int_] | NDArray[np.bool_],
                      state: list[int] = None):
    """{ProbeView#set_captured_electrodes()}"""
    electrodes = self.probe.all_electrodes(self.channelmap)
    captured = [electrodes[int(it)] for it in np.arange(len(self.s))[index]]

    view = controller.get_app().probe_view
    if state is None:
        view.set_captured_electrodes(captured)
    else:
        for s in state:
            try:
                data = view.data_electrodes[s]
            except KeyError:
                pass
            else:
                view.set_captured_electrodes(captured, data)


@doc_link()
def captured_electrodes(controller: ControllerView, all=False) -> NDArray[np.int_]:
    """{ProbeView#get_captured_electrodes_index()}"""
    view = controller.get_app().probe_view
    if all:
        captured = view.get_captured_electrodes_index(None, reset=False)
    else:
        captured = view.get_captured_electrodes_index(view.data_electrodes[ProbeDesp.STATE_USED], reset=False)

    return np.unique(captured)


@doc_link()
def set_state_for_captured(self: BlueprintFunctions, controller: ControllerView,
                           state: int,
                           index: NDArray[np.int_] | NDArray[np.bool_] = None):
    """{ProbeView#set_state_for_captured()}"""
    if index is not None:
        capture_electrode(self, controller, index)
    controller.get_app().probe_view.set_state_for_captured(state)


@doc_link()
def set_category_for_captured(self: BlueprintFunctions, controller: ControllerView,
                              category: int,
                              index: NDArray[np.int_] | NDArray[np.bool_] = None):
    """{ProbeView#set_category_for_captured()}"""
    if index is not None:
        capture_electrode(self, controller, index)
    controller.get_app().probe_view.set_category_for_captured(category)


@doc_link()
def refresh_selection(self: BlueprintFunctions, controller: ControllerView, selector: str = None):
    """{ProbeView#refresh_selection()}"""
    view = controller.get_app().probe_view

    old_select_args = dict(view.selecting_parameters)
    if selector is not None:
        view.selecting_parameters['selector'] = selector

    self.apply_blueprint(view.electrodes)

    try:
        view.refresh_selection()
        view.update_electrode()
        self.set_channelmap(view.channelmap)
    finally:
        view.selecting_parameters = old_select_args


@doc_link()
def atlas_add_label(controller: ControllerView, text: str,
                    pos: tuple[float, float] | tuple[float, float, float],
                    origin: str = 'bregma', *, replace=True):
    """{AtlasBrainView#add_label()}"""
    view: AtlasBrainView
    if (view := controller.get_view('AtlasBrainView')) is None:
        return

    view.add_label(text, pos, origin, replace=replace)


@doc_link()
def atlas_del_label(controller: ControllerView, i: int | str | list[int | str]):
    """{AtlasBrainView#del_label()}"""
    view: AtlasBrainView
    if (view := controller.get_view('AtlasBrainView')) is None:
        return

    match i:
        case int(i):
            ii = [i]
        case list(tmp):
            ii = []
            for it in tmp:
                if isinstance(it, int):
                    ii.append(it)
                elif isinstance(it, str):
                    if (it := view.index_label(it)) is not None:
                        ii.append(it)
                else:
                    raise TypeError()
        case str(text):
            if (it := view.index_label(text)) is None:
                return

            ii = [it]
        case _:
            raise TypeError()

    view.del_label(ii)


@doc_link()
def atlas_clear_labels(controller: ControllerView):
    """{AtlasBrainView#clear_labels()}"""
    view: AtlasBrainView
    if (view := controller.get_view('AtlasBrainView')) is None:
        return
    view.clear_labels()
