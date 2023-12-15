from collections.abc import Callable

from bokeh.layouts import row
from bokeh.models import Button, UIElement

__all__ = ['ButtonFactory', 'col_layout']


class ButtonFactory(object):
    def __init__(self, **kwargs):
        self.__kwargs = kwargs

    def __call__(self, label: str, callback: Callable[..., None], **kwargs) -> Button:
        for k, v in self.__kwargs.items():
            kwargs.setdefault(k, v)
        btn = Button(label=label, **kwargs)
        btn.on_click(callback)
        return btn


def col_layout(model: list[UIElement], n: int) -> list[UIElement]:
    ret = []
    for i in range(0, len(model), n):
        ret.append(row(model[i:i + n]))
    return ret
