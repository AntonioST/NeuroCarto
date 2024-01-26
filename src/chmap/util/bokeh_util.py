import functools
import inspect
from collections.abc import Callable
from typing import Any

from bokeh.layouts import row
from bokeh.models import Button, UIElement, Slider

__all__ = [
    'ButtonFactory',
    'SliderFactory',
    'as_callback',
    'col_layout',
    'is_recursive_called',
]


class ButtonFactory(object):
    def __init__(self, **kwargs):
        self.__kwargs = kwargs

    def __call__(self, label: str, callback: Callable[..., None], **kwargs) -> Button:
        for k, v in self.__kwargs.items():
            kwargs.setdefault(k, v)

        btn = Button(label=label, **kwargs)
        btn.on_click(callback)
        return btn


class SliderFactory(object):
    def __init__(self, **kwargs):
        self.__kwargs = kwargs

    def __call__(self, title: str,
                 slide: tuple[float, float, float] | tuple[float, float, float, float],
                 callback: Callable[..., None], **kwargs) -> Slider:
        for k, v in self.__kwargs.items():
            kwargs.setdefault(k, v)

        match slide:
            case (start, end, step):
                if start <= 0 <= end:
                    value = 0
                else:
                    value = start
            case (start, end, step, value):
                pass
            case _:
                raise TypeError()

        ret = Slider(title=title, start=start, end=end, step=step, value=value, **kwargs)
        ret.on_change('value', as_callback(callback))
        return ret


def as_callback(callback: Callable[..., None], *args, **kwargs) -> Callable[[str, Any, Any], None]:
    if len(args) != 0 or len(kwargs) != 0:
        callback = functools.partial(callback, *args, **kwargs)

    s = inspect.signature(callback)

    match len(s.parameters):
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
    stack = inspect.stack()
    caller = stack[1]

    for i, frame in enumerate(stack[2:]):
        if i < limit and frame.filename == caller.filename and frame.function == caller.function:
            return True
    return False