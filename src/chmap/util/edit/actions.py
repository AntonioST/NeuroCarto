from typing import Any

import numpy as np
from numpy.typing import NDArray

from chmap.probe import ProbeDesp
from chmap.util.util_blueprint import BlueprintFunctions
from chmap.views.base import ViewBase, ControllerView
from chmap.views.data import DataHandler

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
]


def new_channelmap(controller: ControllerView, code: int | str) -> Any:
    app = controller.get_app()
    app.on_new(code)
    return app.probe_view.channelmap


def log_message(controller: ControllerView, *message: str):
    if isinstance(controller, ViewBase):
        controller.log_message(*message)


def set_status_line(controller: ControllerView, message: str, *, decay: float = None):
    if isinstance(controller, ViewBase):
        controller.set_status(message, decay=decay)


def draw(self: BlueprintFunctions, controller: ControllerView,
         a: NDArray[np.float_] | None, *,
         view: str | type[ViewBase] = None):
    if isinstance(controller, DataHandler):
        controller.on_data_update(self.probe, self.probe.all_electrodes(self.channelmap), a)
    elif isinstance(view_target := controller.get_view(view), DataHandler):
        view_target.on_data_update(self.probe, self.probe.all_electrodes(self.channelmap), a)


def capture_electrode(self: BlueprintFunctions, controller: ControllerView,
                      index: NDArray[np.int_] | NDArray[np.bool_],
                      state: list[int] = None):
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


def captured_electrodes(self: BlueprintFunctions, controller: ControllerView,
                        all=False) -> NDArray[np.int_]:
    view = controller.get_app().probe_view
    if all:
        captured = view.get_captured_electrodes_index(None, reset=False)
    else:
        captured = view.get_captured_electrodes_index(view.data_electrodes[ProbeDesp.STATE_USED], reset=False)

    return np.unique(captured)


def set_state_for_captured(self: BlueprintFunctions, controller: ControllerView,
                           state: int,
                           index: NDArray[np.int_] | NDArray[np.bool_] = None):
    if index is not None:
        capture_electrode(self, controller, index)
    controller.get_app().probe_view.set_state_for_captured(state)


def set_category_for_captured(self: BlueprintFunctions, controller: ControllerView,
                              category: int,
                              index: NDArray[np.int_] | NDArray[np.bool_] = None):
    if index is not None:
        capture_electrode(self, controller, index)
    controller.get_app().probe_view.set_category_for_captured(category)


def refresh_selection(self: BlueprintFunctions, controller: ControllerView, selector: str = None):
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
