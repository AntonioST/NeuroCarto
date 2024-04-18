from __future__ import annotations

import inspect
import textwrap
from typing import TYPE_CHECKING

from numpy.typing import NDArray

from neurocarto.util.util_blueprint import BlueprintFunctions
from neurocarto.util.utils import SPHINX_BUILD, doc_link
from neurocarto.views.base import ControllerView

if TYPE_CHECKING:
    from neurocarto.views.blueprint_script import BlueprintScriptView
elif SPHINX_BUILD:
    ViewBase = 'neurocarto.views.base.ViewBase'
    BlueprintScriptView = 'neurocarto.views.blueprint_script.BlueprintScriptView'

__all__ = [
    'print_local',
    'profile_script'
]


def print_local(self: BlueprintFunctions, data: NDArray, i: int, size: int = 1) -> str:
    """
    print electrode data around the electrode *i*.

    :param self:
    :param data: electrode data
    :param i: electrode index
    :param size: local size
    :return: ascii art text.
    """
    s = int(self.s[i])
    x = int(self.x[i] / self.dx)
    y = int(self.y[i] / self.dy)

    ret: list[list[str]] = []
    for dy in range(-size, size + 1):
        row: list[str] = []
        ret.append(row)
        for dx in range(-size, size + 1):
            j = self._position_index.get((s, x + dx, y + dy), None)
            if j is None:
                row.append('_')
            else:
                row.append(str(data[j]))

    width = max([max([len(it) for it in row]) for row in ret])
    fmt = f'%{width}s'
    return '\n'.join(reversed([
        ' '.join([
            fmt % it
            for it in row
        ]) for row in ret
    ]))


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.misc_profile_script.__doc__))
def profile_script(self: BlueprintFunctions, controller: ControllerView, script: str, /, *args, **kwargs):
    """
    {DOC}
    """

    edit: BlueprintScriptView
    if (edit := controller.get_view('BlueprintScriptView')) is None:  # type: ignore[assignment]
        return

    from neurocarto.util.debug import Profiler
    from neurocarto.files import user_cache_file
    dat_file = user_cache_file(controller.get_app().config, f'profile-{script}.dat')

    info = edit.get_script(script)

    edit.logger.debug('profile_script(%s) start', script)

    with Profiler(dat_file) as profile:
        ret = info.script(self, *args, **kwargs)
        if inspect.isgenerator(ret):
            try:
                while True:
                    ret.send(None)
            except StopIteration:
                pass

    if (e := profile.exception) is not None:
        edit.logger.debug('profile_script(%s) fail. spent %.4fs', script, profile.duration, exc_info=e)
    else:
        edit.logger.debug('profile_script(%s) done. spent %.4fs', script, profile.duration)

    profile.print_command()
    if (e := profile.exception) is not None:
        raise e
