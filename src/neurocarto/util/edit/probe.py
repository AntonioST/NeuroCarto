from __future__ import annotations

import textwrap
from typing import Any, Literal

import numpy as np
from neurocarto.probe import ProbeDesp, ElectrodeDesp, M
from neurocarto.util.util_blueprint import BlueprintFunctions
from neurocarto.util.utils import SPHINX_BUILD, doc_link
from neurocarto.views.base import ControllerView
from numpy.typing import NDArray

if SPHINX_BUILD:
    ProbeView = 'neurocarto.views.probe.ProbeView'

__all__ = [
    'new_channelmap',
    'capture_electrode',
    'clear_capture_electrode',
    'captured_electrodes',
    'set_state_for_captured',
    'set_category_for_captured',
    'refresh_selection',
    'select_electrodes',
]


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.new_channelmap.__doc__))
def new_channelmap(controller: ControllerView, code: int | str) -> Any:
    """
    {DOC}
    :see: {BlueprintFunctions#new_channelmap()}
    """
    app = controller.get_app()
    app.on_new(code)
    return app.probe_view.channelmap


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.capture_electrode.__doc__))
def capture_electrode(self: BlueprintFunctions, controller: ControllerView,
                      index: NDArray[np.int_] | NDArray[np.bool_],
                      state: list[int] = None,
                      mode: Literal['replace', 'append', 'exclude'] = 'replace'):
    """
    {DOC}
    :see: {BlueprintFunctions#capture_electrode()}
    """
    electrodes = self.electrodes
    captured = set([int(it) for it in np.arange(len(self.s))[index]])
    previous = set([int(it) for it in captured_electrodes(controller, all=True)])

    match mode:
        case 'replace':
            pass
        case 'append':
            captured.update(previous)
        case 'exclude':
            captured = previous.difference(captured)
        case _:
            raise ValueError(f'unknown mode "{mode}"')

    captured = [electrodes[it] for it in captured]

    view = controller.get_app().probe_view
    if state is None:
        view.set_capture_electrodes(captured)
    else:
        for s in state:
            try:
                data = view.data_electrodes[s]
            except KeyError:
                pass
            else:
                view.set_capture_electrodes(captured, data)


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.clear_capture_electrode.__doc__))
def clear_capture_electrode(controller: ControllerView):
    """
    {DOC}
    :see: {BlueprintFunctions#clear_capture_electrode()}
    """
    controller.get_app().probe_view.clear_capture_electrode()


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.captured_electrodes.__doc__))
def captured_electrodes(controller: ControllerView, all=False) -> NDArray[np.int_]:
    """
    {DOC}
    :see: {BlueprintFunctions#captured_electrodes()}
    """
    view = controller.get_app().probe_view
    if all:
        captured = view.get_captured_electrodes_index(None, reset=False)
    else:
        captured = view.get_captured_electrodes_index(view.data_electrodes[ProbeDesp.STATE_USED], reset=False)

    return np.unique(captured)


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.set_state_for_captured.__doc__))
def set_state_for_captured(self: BlueprintFunctions, controller: ControllerView,
                           state: int,
                           index: NDArray[np.int_] | NDArray[np.bool_] = None):
    """
    {DOC}
    :see: {BlueprintFunctions#set_state_for_captured()}
    """
    if index is not None:
        capture_electrode(self, controller, index)
    controller.get_app().probe_view.set_state_for_captured(state)


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.set_category_for_captured.__doc__))
def set_category_for_captured(self: BlueprintFunctions, controller: ControllerView,
                              category: int,
                              index: NDArray[np.int_] | NDArray[np.bool_] = None):
    """
    {DOC}
    :see: {BlueprintFunctions#set_category_for_captured()}
    """
    if index is not None:
        capture_electrode(self, controller, index)
    controller.get_app().probe_view.set_category_for_captured(category)


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.refresh_selection.__doc__))
def refresh_selection(self: BlueprintFunctions, controller: ControllerView, selector: str = None):
    """
    {DOC}
    :see: {BlueprintFunctions#refresh_selection()}
    """
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
def select_electrodes(self: BlueprintFunctions, chmap: M = None,
                      blueprint: list[ElectrodeDesp] | NDArray[np.int_] = None, **kwargs) -> M:
    """
    Run electrode selection for a channelmap based on the blueprint.

    :param self:
    :param chmap:
    :param blueprint:
    :param kwargs: selector extra parameters
    :return: selection result
    :see: {ProbeDesp#select_electrodes}
    """
    desp = self.probe

    if chmap is None:
        chmap = self.channelmap

    if blueprint is None:
        blueprint = self.blueprint()

    if isinstance(blueprint, np.ndarray):
        blueprint = self.apply_blueprint(blueprint=blueprint)

    return desp.select_electrodes(chmap, blueprint, **kwargs)
