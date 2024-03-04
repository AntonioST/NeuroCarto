from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from chmap.util.util_blueprint import BlueprintFunctions
from chmap.util.utils import SPHINX_BUILD, doc_link
from chmap.views.base import ViewBase, ControllerView
from chmap.views.data import DataHandler

if TYPE_CHECKING:
    from chmap.views.edit_blueprint import BlueprintScriptView
elif SPHINX_BUILD:
    ViewBase = 'chmap.views.base.ViewBase'
    BlueprintScriptView = 'chmap.views.edit_blueprint.BlueprintScriptView'

__all__ = [
    'log_message',
    'set_status_line',
    'draw',
    'has_script',
    'call_script',
]


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
def has_script(controller: ControllerView, script: str) -> bool:
    """{BlueprintScriptView#run_script()}"""
    edit: BlueprintScriptView = controller.get_view('BlueprintScriptView')
    if edit is None:
        return False
    try:
        edit.get_script(script)
    except ImportError:
        return False

    return True


@doc_link()
def call_script(self: BlueprintFunctions, controller: ControllerView, script: str, /, *args, **kwargs):
    """{BlueprintScriptView#run_script()}"""
    edit: BlueprintScriptView = controller.get_view('BlueprintScriptView')
    if edit is None:
        return

    # TODO logging
    info = edit.get_script(script)
    if inspect.isgenerator(ret := info.script(self, *args, **kwargs)):
        edit._run_script_generator(self, script, ret)
