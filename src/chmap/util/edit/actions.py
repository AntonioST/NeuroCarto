from __future__ import annotations

import inspect
import time
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from chmap.util.util_blueprint import BlueprintFunctions
from chmap.util.utils import SPHINX_BUILD, doc_link
from chmap.views.base import ViewBase, ControllerView
from chmap.views.data import DataHandler

if TYPE_CHECKING:
    from chmap.views.blueprint_script import BlueprintScriptView
elif SPHINX_BUILD:
    ViewBase = 'chmap.views.base.ViewBase'
    BlueprintScriptView = 'chmap.views.blueprint_script.BlueprintScriptView'

__all__ = [
    'log_message',
    'set_status_line',
    'draw',
    'has_script',
    'call_script',
    'interrupt_script',
    'profile_script',
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
    except (ImportError, TypeError):
        return False

    return True


@doc_link()
def call_script(self: BlueprintFunctions, controller: ControllerView, script: str, /, *args, **kwargs):
    """{BlueprintScriptView#run_script()}"""
    edit: BlueprintScriptView = controller.get_view('BlueprintScriptView')
    if edit is None:
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
    edit: BlueprintScriptView = controller.get_view('BlueprintScriptView')
    if edit is None:
        return False

    try:
        edit.interrupt_script(script)
    except ValueError:
        return False
    else:
        return True


@doc_link()
def profile_script(self: BlueprintFunctions, controller: ControllerView, script: str, /, *args, **kwargs):
    """{BlueprintScriptView#run_script()}"""

    edit: BlueprintScriptView = controller.get_view('BlueprintScriptView')
    if edit is None:
        return

    import cProfile

    info = edit.get_script(script)
    profile = cProfile.Profile()

    try:
        edit.logger.debug('profile_script(%s)', script)

        t = time.time()
        profile.enable()
        try:
            ret = info.script(self, *args, **kwargs)
            if inspect.isgenerator(ret):
                try:
                    while True:
                        ret.send(None)
                except StopIteration:
                    pass
        finally:
            profile.disable()
            t = time.time() - t

        edit.logger.debug('profile_script(%s) done. spent %.4fs', script, t)
        _save_profile_data(controller, script, profile)
    except BaseException as e:
        edit.logger.debug('profile_script(%s) fail. spent %.4fs', script, t, exc_info=e)
        _save_profile_data(controller, script, profile)
        raise e


def _save_profile_data(controller: ControllerView, script: str, profile):
    from chmap.files import user_cache_file
    dat_file = user_cache_file(controller.get_app().config, f'profile-{script}.dat')
    print(f'save {dat_file}')
    profile.dump_stats(dat_file)
    print(f'python -m gprof2dot -f pstats {dat_file} | dot -T png -o profile-{script}.png')
