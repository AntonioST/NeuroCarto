from __future__ import annotations

import inspect
import textwrap
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurocarto.util.util_blueprint import BlueprintFunctions
from neurocarto.util.utils import SPHINX_BUILD, doc_link
from neurocarto.views.base import ViewBase, ControllerView

if TYPE_CHECKING:
    from neurocarto.views.blueprint_script import BlueprintScriptView
elif SPHINX_BUILD:
    ViewBase = 'neurocarto.views.base.ViewBase'
    BlueprintScriptView = 'neurocarto.views.blueprint_script.BlueprintScriptView'

__all__ = [
    'log_message',
    'set_status_line',
    'draw',
    'has_script',
    'call_script',
    'interrupt_script',
    'set_script_input',
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
         a: NDArray[np.float_] | None):
    """{BlueprintScriptView#on_data_update()}"""
    if a is not None and len(a) != len(self.s):
        raise ValueError('length mismatch')

    edit: BlueprintScriptView
    if (edit := controller.get_view('BlueprintScriptView')) is None:  # type: ignore[assignment]
        return

    edit.on_data_update(self.probe, a)


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.has_script.__doc__))
def has_script(controller: ControllerView, script: str) -> bool:
    """
    {DOC}
    """
    edit: BlueprintScriptView
    if (edit := controller.get_view('BlueprintScriptView')) is None:  # type: ignore[assignment]
        return False

    try:
        edit.get_script(script)
    except (ImportError, TypeError):
        return False

    return True


@doc_link()
def call_script(self: BlueprintFunctions, controller: ControllerView, script: str, /, *args, **kwargs):
    """{BlueprintScriptView#run_script()}"""
    edit: BlueprintScriptView
    if (edit := controller.get_view('BlueprintScriptView')) is None:  # type: ignore[assignment]
        return

    info = edit.get_script(script)

    edit.logger.debug('call_script(%s)', script)
    try:
        ret = info.script(self, *args, **kwargs)
    except BaseException as e:
        edit.logger.debug('call_script(%s) fail', script, exc_info=e)
        raise e
    else:
        if inspect.isgenerator(ret):
            edit.logger.debug('call_script(%s) return generator', script)
            edit._run_script_generator(self, script, ret)
        else:
            edit.logger.debug('call_script(%s) done', script)


@doc_link()
def interrupt_script(controller: ControllerView, script: str) -> bool:
    """{BlueprintScriptView#interrupt_script()}"""
    edit: BlueprintScriptView
    if (edit := controller.get_view('BlueprintScriptView')) is None:  # type: ignore[assignment]
        return False

    try:
        edit.interrupt_script(script)
    except ValueError:
        return False
    else:
        return True


def set_script_input(controller: ControllerView, script: str | None, *text: str | None):
    edit: BlueprintScriptView
    if (edit := controller.get_view('BlueprintScriptView')) is None:  # type: ignore[assignment]
        return False

    if script is None:
        script = edit.script_select.value

    script_input = ','.join(filter(lambda it: it is not None, text))

    if script == edit.script_select.value:
        edit.script_input.value_input = script_input
    elif script in edit.actions:
        edit._script_input_cache[script] = script_input

