from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Any

from neurocarto.probe import ProbeDesp, get_probe_desp
from neurocarto.util.utils import SPHINX_BUILD, doc_link

if TYPE_CHECKING:
    from neurocarto.util.util_blueprint import BlueprintFunctions
    from neurocarto.views.base import ViewBase
elif SPHINX_BUILD:
    ViewBase = 'neurocarto.views.base.ViewBase'
    BlueprintFunctions = 'neurocarto.util.util_blueprint.BlueprintFunctions'
    BlueprintScript = 'neurocarto.util.edit.script.BlueprintScript',
    BlueprintScriptView = 'neurocarto.views.blueprint_script.BlueprintScriptView',

__all__ = [
    #
    'use_probe',
    'check_probe',
    'get_use_probe',
    'RequestChannelmapType',
    'RequestChannelmapTypeError',
    #
    'use_view',
    'get_use_view',
    'RequestView',
]


@doc_link()
class RequestChannelmapType(NamedTuple):
    """
    An annotation data to indicate the probe request for a blueprint script.

    **Do not create by yourself**. In order to use this, use {use_probe()} and {get_use_probe()} instead.
    """

    probe: str | type[ProbeDesp] | None
    """request probe name or type."""

    code: int | None
    """request channelmap code."""

    create: bool = True
    """create a channelmap when it is missing."""

    check: bool = True
    """check it is existed and match to the request. raise error when mis-match"""

    @property
    def probe_name(self) -> str:
        """Probe name"""
        if isinstance(self.probe, type):
            return self.probe.__name__

        elif isinstance(self.probe, str):
            return self.probe

        else:
            raise RuntimeError()

    def match_probe(self, probe: ProbeDesp | None, chmap: Any | None = None) -> bool:
        """
        Does *probe* and *chmap* match this request?

        :param probe:
        :param chmap:
        :return:
        """
        if probe is None:
            return False

        if isinstance(self.probe, type):
            if not isinstance(probe, self.probe):
                return False

        elif isinstance(self.probe, str):
            if type(probe).__name__ != self.probe:
                try:
                    probe_type = get_probe_desp(self.probe)
                except BaseException:
                    return False
                else:
                    if not isinstance(probe, probe_type):
                        return False

        else:
            raise TypeError()

        if self.code is None or chmap is None:
            return True

        if (code := probe.channelmap_code(chmap)) is None:
            return False
        return self.code == code


@doc_link()
def use_probe(probe: str | type[ProbeDesp] = ProbeDesp, code: int = None, *,
              create: bool = None, check=True):
    """
    Decorate a blueprint script ({BlueprintScript}) to indicate this function
    request a particular probe type.

    .. code-block:: python

        @use_probe('npx', 24) # checking and creating probe if necessary
        def my_probe(bp: BlueprintFunctions):
            bp.check_probe('npx', 24) # no need anymore in common case.

    It also allows {BlueprintScriptView} to filter suitable scripts for a probe.

    :param probe: probe type or its name. It is omitted, just check a probe is existed.
    :param code: channelmap code
    :param create: create the probe if there is no probe in the main figure.
        If ``True``, it requires *code* should be not ``None``.
        Force ``False`` if *probe* is omitted.
    :param check: check the current probe type automatically before entering the script.
    :see: {get_use_probe()}
    :see: {BlueprintFunctions#check_probe()}
    """
    if probe is None:
        raise ValueError('NoneType probe')

    if probe is ProbeDesp:
        create = False
        code = None

    if create is None:
        create = code is not None

    if create and code is None:
        raise ValueError('create mode need non-None code')

    def _decorator(func):
        func.__neurocarto_checking_use_probe__ = RequestChannelmapType(probe, code, create, check)
        return func

    return _decorator


@doc_link()
def get_use_probe(func) -> RequestChannelmapType | None:
    """
    Get {RequestChannelmapType}.

    :param func: blueprint script function
    :return:
    """
    return getattr(func, '__neurocarto_checking_use_probe__', None)


class RequestChannelmapTypeError(RuntimeError):
    def __init__(self, request: RequestChannelmapType):
        """

        :param request: request probe type.
        """
        self.request = request

        if (probe := request.probe) is None:
            message = 'Require a probe'
        elif isinstance(probe, str):
            message = f'Request Probe[{probe}]'
        else:
            message = f'Request {probe.__name__}'

        if (chmap_code := request.code) is not None:
            message += f'[{chmap_code}].'
        else:
            message += '.'

        super().__init__(message)


@doc_link()
def check_probe(self: BlueprintFunctions,
                probe: str | type[ProbeDesp] | RequestChannelmapType | None = None,
                code: int = None):
    """
    check request probe type and channelmap code.

    :param self:
    :param probe: request probe type. It could be family name (via {get_probe_desp()}), {ProbeDesp} type or class name.
        It ``None``, checking a probe has created, and its type doesn't matter.
    :param code: request channelmap code
    :raise RequestChannelmapTypeError: when check failed.
    """
    if isinstance(probe, RequestChannelmapType):
        request = probe
    else:
        request = RequestChannelmapType(probe, code)

    current_probe = self.probe
    current_chmap = self.channelmap

    if probe is None:
        if current_chmap is None:
            raise RequestChannelmapTypeError(request)
        return

    if not request.match_probe(current_probe, current_chmap):
        raise RequestChannelmapTypeError(request)


@doc_link()
class RequestView(NamedTuple):
    """
    An annotation data to indicate the view dependency for a blueprint script.

    **Do not create by yourself**. In order to use this, use {use_view()} and {get_use_view()} instead.
    """

    view_type: str | type[ViewBase]
    """request view name or type."""


@doc_link()
def use_view(view: str | type[ViewBase]):
    """
    Decorate a blueprint script ({BlueprintScript}) to indicate this function
    request a particular {ViewBase} to work.

    .. code-block:: python

        @use_view('AtlasView')
        def my_function(bp: BlueprintFunctions):
            if (atlas := bp.use_view('AtlasView')) is None:
                raise RuntimeError('AtlasView is not loaded')

    It also allows {BlueprintScriptView} to filter suitable scripts, which depends on
    existed views.

    :param view: request view name or type.
    :return:
    :see: {get_use_view()}
    :see: {BlueprintFunctions#use_view()}
    """

    def _decorator(func):
        func.__neurocarto_checking_use_view__ = RequestView(view)
        return func

    return _decorator


@doc_link()
def get_use_view(func) -> RequestView | None:
    """
    Get {RequestView}.

    :param func: blueprint script function
    :return:
    """
    return getattr(func, '__neurocarto_checking_use_view__', None)
