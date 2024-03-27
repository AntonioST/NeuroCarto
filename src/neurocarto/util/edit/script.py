from __future__ import annotations

import collections
import functools
import html
import inspect
import re
import sys
import textwrap
from pathlib import Path
from typing import Protocol, TYPE_CHECKING, cast, NamedTuple, Any

from neurocarto.util.util_blueprint import BlueprintFunctions
from neurocarto.util.utils import import_name, get_import_file, doc_link

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

if TYPE_CHECKING:
    from neurocarto.util.edit.checking import RequestChannelmapType, RequestView

__all__ = [
    'BlueprintScript',
    'BlueprintScriptInfo',
    'format_html_doc',
    'script_html_doc',
]

EXAMPLE_DOCUMENT = """\
Document here.

:param bp:
:param a0: (type=default) parameter description
:param a1: (int=0) as example.
"""


def format_html_doc(doc: str) -> str:
    ret = html.escape(doc.strip())
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
    __name__: str

    @doc_link(
        use_probe='neurocarto.util.edit.checking.use_probe',
        use_view='neurocarto.util.edit.checking.use_view',
    )
    def __call__(self, bp: BlueprintFunctions, *args, **kwargs) -> None:
        """
        **Command structure**

        .. code-block:: python

            def example_script(bp: BlueprintFunctions):
                \"""Document\"""
                bp.check_probe() # checking used probe
                ...

        **Available decorators**

        .. hlist::
            :columns: 2

            * {use_probe()}
            * {use_view()}

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

        Note that we do not fetch the typing information and default value via reflection.
        (only parameters' name. Others would happen in the future, probability)
        The creator has to manually type those information in the document.

        :param bp: script running context.
        :param args:
        :param kwargs:
        """
        pass


@doc_link()
class BlueprintScriptInfo(NamedTuple):
    """A {BlueprintScript} holder"""

    name: str
    """script name."""

    module: str | None
    """script module path expression in 'MODUlE:NAME'"""

    filepath: Path | None
    """actual module file path"""

    time_stamp: float | None
    """module file last modified time."""

    script: BlueprintScript
    """script instance."""

    @classmethod
    def load(cls, name: str, module_path: str) -> Self:
        """
        Load a blueprint script.

        :param name: script name.
        :param module_path: script module path.
        :return:
        """
        script = cast(BlueprintScript, import_name('blueprint script', module_path))
        if not callable(script):
            raise ImportError(f'script {name} not callable')

        script_file = get_import_file(module_path)
        time_stamp = None if script_file is None or not script_file.exists() else script_file.stat().st_mtime
        return BlueprintScriptInfo(name, module_path, script_file, time_stamp, script)  # type: ignore[return-value]

    def check_changed(self) -> bool:
        """
        Was source module file of the script modified from latest loading?

        :return: True when it has modified at this moment.
        """
        if self.filepath is None or self.time_stamp is None:
            return False
        if not self.filepath.exists():
            return True
        t = self.filepath.stat().st_mtime
        return self.time_stamp < t

    def reload(self) -> Self:
        """
        Reload the script from source module file.

        :return:
        """
        if self.module is None:
            raise ImportError()

        script = cast(BlueprintScript, import_name('blueprint script', self.module, reload=True))
        if not callable(script):
            raise ImportError(f'script {self.name} not callable')

        script_file = self.filepath
        time_stamp = None if script_file is None or not script_file.exists() else script_file.stat().st_mtime
        return self._replace(time_stamp=time_stamp, script=script)

    @doc_link(use_probe='neurocarto.util.edit.checking.use_probe')
    def script_use_probe(self) -> RequestChannelmapType | None:
        """
        Get the probe requirement.

        :return:
        :see: {use_probe()}
        """
        from neurocarto.util.edit.checking import get_use_probe
        return get_use_probe(self.script)

    @doc_link(use_view='neurocarto.util.edit.checking.use_view')
    def script_use_view(self) -> RequestView | None:
        """
        Get the view requirement

        :return:
        :see: {use_view()}
        """
        from neurocarto.util.edit.checking import get_use_view
        return get_use_view(self.script)

    def eval(self, bp: BlueprintFunctions, script_input: str) -> Any:
        """
        Eval *script_input* and invoke actual script function.

        Although *script_input* should be a valid Python code,
        we cheat on the name resolution that take the undefined variable as a str by its name.
        For example, the following *script_input* will be considered::

            (input) a,1,"b,3"
            (output) ("a", 1, "b,3")

        There are some limitations:

        * This function does not do the argument type validation.
        * If it contains operator, such as '.' in filename, it fails.
        * for above case, use "" to quote them.

        :param bp:
        :param script_input:
        :return: blueprint script function's return
        """

        class Missing(collections.defaultdict):
            def __missing__(self, key):
                return key

        return eval(f'__script_func__({script_input})', {}, Missing(__script_func__=functools.partial(self.script, bp)))

    def __call__(self, bp: BlueprintFunctions, *args, **kwargs) -> Any:
        """
        Invoke the script function.

        :param bp:
        :param args: script function's positional arguments.
        :param kwargs: script function's keyword arguments.
        :return: script function's return.
        """
        return self.script(bp, *args, **kwargs)


def script_signature(script: BlueprintScriptInfo) -> str:
    s = inspect.signature(script.script)

    name = script.script.__name__

    p = []
    for i, it in enumerate(s.parameters):
        if i != 0:
            match s.parameters[it].kind:
                case inspect.Parameter.VAR_POSITIONAL:
                    p.append(f'*{it}')
                case inspect.Parameter.VAR_KEYWORD:
                    p.append(f'**{it}')
                case inspect.Parameter.KEYWORD_ONLY:
                    p.append(f'{it}=')
                case _:
                    p.append(it)

    p = ', '.join(p)
    return f'{name}({p})'


def script_doc(script: BlueprintScriptInfo, html=False) -> str | None:
    if (doc := script.script.__doc__) is not None:
        ret = textwrap.dedent(doc)
        if html:
            ret = format_html_doc(ret)
        return ret
    return None


def script_html_doc(script: BlueprintScriptInfo) -> str:
    head = script_signature(script)
    if (doc := script_doc(script, html=True)) is None:
        return f"""
    <div style="padding-left: 2em">
        <p><b>{head}</b></p>
    </div>
"""
    else:
        return f"""
    <div style="padding-left: 2em">
        <style type="text/css">
            p.carto-script-head+div.carto-script-doc {{
                display: none;
            }}
            p.carto-script-head:hover+div.carto-script-doc {{
                display: block;
            }}
        </style>
        <p class="carto-script-head"><b>{head}</b></p>
        <div class="carto-script-doc" style="padding-left: 1em">{doc}</div>
    </div>
"""
