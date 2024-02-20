from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Any

from chmap.probe import ProbeDesp, get_probe_desp
from chmap.util.utils import SPHINX_BUILD, doc_link

if TYPE_CHECKING:
    from chmap.util.util_blueprint import BlueprintFunctions
elif SPHINX_BUILD:
    BlueprintFunctions = 'chmap.util.util_blueprint.BlueprintFunctions'

__all__ = [
    'use_probe',
    'check_probe',
    'get_use_probe',
    'RequestChannelmapTypeRequest',
    'RequestChannelmapTypeError',
]


@doc_link()
class RequestChannelmapTypeRequest(NamedTuple):
    """
    **Do not create by yourself**. In order to use this, use {use_probe()} and {get_use_probe()} instead.

    An annotation to indicate a blueprint script's probe request.
    """

    probe: str | type[ProbeDesp] | None
    code: int | None
    create: bool = True
    check: bool = True

    @property
    def probe_name(self) -> str:
        """Probe name"""
        if isinstance(self.probe, type):
            return self.probe.__name__

        elif isinstance(self.probe, str):
            return self.probe

        else:
            raise RuntimeError()

    def match_probe(self, probe: ProbeDesp, chmap: Any | None = None) -> bool:
        """
        Does *probe* and *chmap* match this request?

        :param probe:
        :param chmap:
        :return:
        """
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


@doc_link(
    BlueprintScript='chmap.views.edit_blueprint.BlueprintScript',
    BlueprintScriptView='chmap.views.edit_blueprint.BlueprintScriptView',
)
def use_probe(probe: str | type[ProbeDesp], code: int = None, *,
              create: bool = None, check=True):
    """
    Decorate a blueprint script ({BlueprintScript}) to indicate this function
    request a particular probe.

    .. code-block:: python

        @use_probe('npx', 24) # checking and creating probe if necessary
        def my_probe(bp: BlueprintFunctions):
            bp.check_probe('npx', 24) # no need anymore in common case.

    If also allow {BlueprintScriptView} to filter suitable script.

    :param probe: probe type or its name
    :param code: channelmap code
    :param create: create the probe if there is no probe in the main figure.
        It `True`, it requires *code* is not `None`.
    :param check: check the current probe type automatically before entering the script.
    """
    if probe is None:
        raise ValueError('NoneType probe')

    if create is None:
        create = code is not None

    if create and code is None:
        raise ValueError('create mode need non-None code')

    def _decorator(func):
        func.__chmap_checking_use_probe__ = RequestChannelmapTypeRequest(probe, code, create, check)
        return func

    return _decorator


def get_use_probe(func) -> RequestChannelmapTypeRequest | None:
    """
    Get {RequestChannelmapTypeRequest}.

    :param func: blueprint script function
    :return:
    """
    return getattr(func, '__chmap_checking_use_probe__', None)


class RequestChannelmapTypeError(RuntimeError):
    def __init__(self, request: RequestChannelmapTypeRequest):
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
                probe: str | type[ProbeDesp] | RequestChannelmapTypeRequest | None = None,
                code: int = None):
    """
    check request probe type and channelmap code.

    :param self:
    :param probe: request probe type. It could be family name (via {get_probe_desp()}), {ProbeDesp} type or class name.
        It `None`, checking a probe has created, and its type doesn't matter.
    :param code: request channelmap code
    :raise RequestChannelmapTypeError: when check failed.
    """
    if isinstance(probe, RequestChannelmapTypeRequest):
        request = probe
    else:
        request = RequestChannelmapTypeRequest(probe, code)

    current_probe = self.probe
    current_chmap = self.channelmap

    if probe is None:
        if current_chmap is None:
            raise RequestChannelmapTypeError(request)
        return

    if not request.match_probe(current_probe, current_chmap):
        raise RequestChannelmapTypeError(request)
