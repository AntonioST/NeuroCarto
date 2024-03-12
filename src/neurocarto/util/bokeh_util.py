import functools
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

from bokeh.layouts import row
from bokeh.models import Button, UIElement, Slider, AutocompleteInput, Tooltip, HelpButton

__all__ = [
    'ButtonFactory',
    'SliderFactory',
    'PathAutocompleteInput',
    'as_callback',
    'col_layout',
    'is_recursive_called',
    'new_help_button'
]


class ButtonFactory(object):
    def __init__(self, **kwargs):
        self.__kwargs = kwargs

    def __call__(self, label: str, callback: Callable[..., None] = None, **kwargs) -> Button:
        for k, v in self.__kwargs.items():
            kwargs.setdefault(k, v)

        btn = Button(label=label, **kwargs)
        if callback is not None:
            btn.on_click(callback)
        return btn


class SliderFactory(object):
    def __init__(self, **kwargs):
        self.__kwargs = kwargs

    def __call__(self, title: str,
                 slide: tuple[float, float, float] | tuple[float, float, float, float],
                 callback: Callable[..., None] = None, **kwargs) -> Slider:
        """

        :param title:
        :param slide: (start, end. step, value?)
        :param callback:
        :param kwargs:
        :return:
        """
        for k, v in self.__kwargs.items():
            kwargs.setdefault(k, v)

        match slide:
            case (start, end, step):
                if start <= 0 <= end:
                    value = type(start)(0)
                else:
                    value = start
            case (start, end, step, value):
                pass
            case _:
                raise TypeError()

        ret = Slider(title=title, start=start, end=end, step=step, value=value, **kwargs)
        if callback is not None:
            ret.on_change('value', as_callback(callback))
        return ret


def as_callback(callback: Callable[..., None], *args, **kwargs) -> Callable[[str, Any, Any], None]:
    if len(args) != 0 or len(kwargs) != 0:
        callback = functools.partial(callback, *args, **kwargs)

    s = inspect.signature(callback)
    p = [
        it for it in s.parameters.values()
        if it.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD and it.default is inspect.Parameter.empty
    ]

    match len(p):
        case 0:
            # noinspection PyUnusedLocal
            def _callback(prop, old, value):
                callback()
        case 1:
            # noinspection PyUnusedLocal
            def _callback(prop, old, value):
                callback(value)
        case 2:
            # noinspection PyUnusedLocal
            def _callback(prop, old, value):
                callback(old, value)
        case 3:
            _callback = callback
        case _:
            raise TypeError()

    return _callback


def col_layout(model: list[UIElement], n: int) -> list[UIElement]:
    ret = []
    for i in range(0, len(model), n):
        ret.append(row(model[i:i + n]))
    return ret


def is_recursive_called(limit=100) -> bool:
    """

    Limitation:

    * check recursive on override methods in same file.

    XXX: does it work?
    https://docs.bokeh.org/en/latest/docs/reference/models/callbacks.html#bokeh.models.Callback.set_from_json

    :param limit:
    :return:
    """
    stack = inspect.stack()
    caller = stack[1]

    for i, frame in enumerate(stack[2:]):
        if i < limit and frame.filename == caller.filename and frame.function == caller.function:
            return True
    return False


class PathAutocompleteInput:
    """
    An alternative of FileInput that allow access full filepath from user's computer.

    Due to FileInput doesn't provide full path because of browser's security reasons,
    so we use AutocompleteInput provide plain text input with auto complete.

    """
    input: AutocompleteInput

    def __init__(self, root: Path,
                 callback: Callable[[Path | None], None] = None,
                 mode: Literal['path', 'dir', 'file'] = 'path',
                 accept: list[str] = None,
                 min_characters=0,
                 max_completions=10,
                 case_sensitive=False,
                 restrict=False,
                 **kwarg):
        """

        :param root:
        :param callback:
        :param mode:
        :param accept: accept file suffix ('.*') or mime type ('*/*') when mode == 'file'
        :param min_characters:
        :param max_completions:
        :param case_sensitive:
        :param restrict:
        :param kwarg:
        """
        if not root.is_dir():
            raise NotADirectoryError()

        self.input = AutocompleteInput(
            min_characters=min_characters,
            max_completions=max_completions,
            case_sensitive=case_sensitive,
            restrict=restrict,
            **kwarg
        )

        self.input.on_change('value_input', as_callback(self._on_typing))
        self.input.on_change('value', as_callback(self._on_enter))

        self._root = root.absolute()
        self._mode = mode
        self._accept = accept
        self._path: Path | None = None
        self._complete_root: Path | None = None
        self._callback: Callable[[Path | None], None] | None = callback

        self._set_complete(self._root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def mode(self) -> Literal['path', 'dir', 'file']:
        return self._mode

    @property
    def value(self) -> str:
        return self.input.value

    @value.setter
    def value(self, value: str):
        self.input.value = value

    @property
    def path(self) -> Path | None:
        return self._path

    @path.setter
    def path(self, path: Path):
        self.input.value = str(path.relative_to(self._root))

        if path.is_dir():
            self._set_complete(path)
        elif (d := path.parent).exists():
            self._set_complete(d)

    def _on_typing(self, value: str):
        f = self._root / value

        if f.is_dir():
            self._set_complete(f)
        elif (d := f.parent).exists():
            self._set_complete(d)

    def _set_complete(self, d: Path):
        if self._complete_root == d:
            return

        self._complete_root = d

        # remove hidden
        fs = [it for it in d.iterdir() if not it.name.startswith('.')]

        match self._mode:
            case 'path':
                pass
            case 'file':
                fs = [
                    it for it in fs
                    if it.is_dir() or (it.is_file() and self._is_accepted(it))
                ]
            case 'dir':
                fs = [it for it in fs if it.is_dir()]
            case _:
                fs = []

        try:
            self.input.completions = [str(it.relative_to(self._root)) + ('/' if it.is_dir() else '') for it in fs]
        except ValueError:
            self.input.completions = []

    def _on_enter(self, value: str):
        from neurocarto.util.bokeh_app import run_later
        if value == '':
            self._path = None
            if self._callback is not None:
                run_later(self._callback, None)
        else:
            f = self._root / value
            self._path = f

            if f.exists() and self._callback is not None:
                match self._mode:
                    case 'path':
                        run_later(self._callback, f)
                    case 'file' if f.is_file() and self._is_accepted(f):
                        run_later(self._callback, f)
                    case 'dir' if f.is_dir():
                        run_later(self._callback, f)

    def _is_accepted(self, f: Path) -> bool:
        if self._accept is None:
            return True

        import mimetypes
        mt, _ = mimetypes.guess_type(f)

        for accept in self._accept:
            if accept.startswith('.'):
                if f.suffix == accept:
                    return True
            elif accept.endswith('/*'):
                if mt is not None and mt.startswith(accept[:-1]):
                    return True
            elif '/' in accept:
                if mt is not None and accept == mt:
                    return True

        return False


def new_help_button(content: str, *, position: str = 'right') -> HelpButton:
    return HelpButton(
        tooltip=Tooltip(content=content, position=position),
        stylesheets=['button.bk-btn {padding:0;}']
    )
