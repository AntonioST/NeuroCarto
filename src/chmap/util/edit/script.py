from __future__ import annotations

import collections
import functools
import inspect
import re
import sys
import textwrap
from pathlib import Path
from typing import Protocol, TYPE_CHECKING, cast, NamedTuple

from chmap.util.util_blueprint import BlueprintFunctions
from chmap.util.utils import import_name, get_import_file, doc_link

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from chmap.util.edit.checking import RequestChannelmapTypeRequest

__all__ = [
    'BlueprintScript',
    'BlueprintScriptInfo',
    'format_html_doc',
]

EXAMPLE_DOCUMENT = """\
Document here.

:param bp:
:param a0: (type=default) parameter description
:param a1: (int=0) as example.
"""


def format_html_doc(doc: str) -> str:
    ret = doc.strip()
    ret = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', ret)
    ret = re.sub(r'\*(.+?)\*', r'<em>\1</em>', ret)
    ret = re.sub(r':param\s+bp:.*?\n?', '\n', ret)
    ret = re.sub(r'(?<=\n):param\s+(\w+):', r'<b>\1</b>:', ret)
    ret = re.sub(r'\n +', '</p><p style="text-indent:2em;">', ret)
    ret = re.sub(r'\n\n', '</p><br/><p>', ret)
    ret = '<p>' + ret.replace('\n', '</p><p>') + '</p>'
    return ret


EXAMPLE_DOCUMENT_HTML = format_html_doc(EXAMPLE_DOCUMENT)


class BlueprintScript(Protocol):
    """A protocol class  to represent a blueprint script function."""

    @doc_link(use_probe='chmap.util.edit.checking.use_probe')
    def __call__(self, bp: BlueprintFunctions, *args, **kwargs) -> None:
        """
        **Command structure**

        .. code-block:: python

            def example_script(bp: BlueprintFunctions):
                \"""Document\"""
                bp.check_probe() # checking used probe
                ...

        **Available decorators**

        {use_probe()}

        **Generator script**

        Blueprint script function could be a generator by using ``yield`` keyword.
        Yielding a number to indicate the caller to wait for a given seconds.
        During running, the "Run" button will become "Stop", in order to interrupt the
        running.

        When interrupted, a ``KeyboardInterrupt`` will be raised. Blueprint script function
        can capture it for cleanup purpose, and then reraise it to the caller to let it
        handle the interrupting message.
        Otherwise, the caller will considered this script has a normal exit.

        .. code-block:: python

            def generator_example_script(bp: BlueprintFunctions):
                yield 10 # wait 10 seconds
                yield # wait until next update tick

                try:
                    yield 100
                except KeyboardInterrupt:
                    ... # do something cleanup
                    raise

        **Document script**

        We use reStructuredText format by default, so we format the document into html
        based on that rules.

        .. code-block:: python

            def example_script(bp: BlueprintFunctions, a0: str, a1:int=0):
                \"""
                {EXAMPLE_DOCUMENT}
                \"""

        * The line ``:param bp:`` will be removed.
        * The line ``:param PARA:`` will be replaced as a bold font (``<b>``).
        * The word ``**WORD**`` will be replaced as a bold font (``<b>``).
        * The word ``*WORD*`` will be replaced as an italic font (``<em>``).

        And it will look like:

        .. raw:: html

            <div class="highlight" style="padding: 10px;"><div style="font-family: monospace;">
            <p><b>example_script(a0, a1)</b></p>
            {EXAMPLE_DOCUMENT_HTML}
            </div></div>

        :param bp: script running context.
        :param args:
        :param kwargs:
        """
        pass


class BlueprintScriptInfo(NamedTuple):
    name: str
    module: str | None  # 'MODUlE:NAME'
    filepath: Path | None
    time_stamp: float | None
    script: BlueprintScript

    @classmethod
    def load(cls, name: str, module_path: str) -> Self:
        script = cast(BlueprintScript, import_name('blueprint script', module_path))
        if not callable(script):
            raise ImportError(f'script {name} not callable')

        script_file = get_import_file(module_path)
        time_stamp = None if script_file is None or not script_file.exists() else script_file.stat().st_mtime
        return BlueprintScriptInfo(name, module_path, script_file, time_stamp, script)

    def check_changed(self) -> bool:
        if self.filepath is None or self.time_stamp is None:
            return False
        if not self.filepath.exists():
            return True
        t = self.filepath.stat().st_mtime
        return self.time_stamp < t

    def reload(self) -> Self:
        if self.module is None:
            raise ImportError()

        script = cast(BlueprintScript, import_name('blueprint script', self.module, reload=True))
        if not callable(script):
            raise ImportError(f'script {self.name} not callable')

        script_file = self.filepath
        time_stamp = None if script_file is None or not script_file.exists() else script_file.stat().st_mtime
        return self._replace(time_stamp=time_stamp, script=script)

    def script_name(self) -> str:
        return self.script.__name__

    def script_signature(self) -> str:
        name = self.script.__name__
        p = ', '.join(self.script_parameters())
        return f'{name}({p})'

    def script_parameters(self) -> list[str]:
        s = inspect.signature(self.script)
        return [it for i, it in enumerate(s.parameters) if i != 0]

    def script_doc(self, html=False) -> str | None:
        if (doc := self.script.__doc__) is not None:
            ret = textwrap.dedent(doc)
            if html:
                ret = format_html_doc(ret)
            return ret
        return None

    def script_use_probe(self) -> RequestChannelmapTypeRequest | None:
        from chmap.util.edit.checking import get_use_probe
        return get_use_probe(self.script)

    def __call__(self, bp: BlueprintFunctions, script_input: str):
        """
        Eval *script_input* and call actual script function.

        Although *script_input* should be a valid Python code,
        we cheat the undefined variable name as a str.
        For example, the following *script_input* will be considered::

            (input) a,1,"b,3"
            (output) ("a", 1, "b,3")

        This function does not do the argument type validation.

        :param bp:
        :param script_input:
        :return: blueprint script function's return
        """

        class Missing(collections.defaultdict):
            def __missing__(self, key):
                return key

        return eval(f'__script_func__({script_input})', {}, Missing(__script_func__=functools.partial(self.script, bp)))
