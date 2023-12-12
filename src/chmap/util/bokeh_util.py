from collections.abc import Callable

from bokeh.models import Div, Button, Select

__all__ = [
    'bold',
    'ButtonFactory',
    'SelectFactory'
]
def bold(text: str) -> Div:
    return Div(text=f'<b>{text}</b>')


class ButtonFactory(object):
    def __init__(self, **kwargs):
        self.__kwargs = kwargs

    def __call__(self, label: str, callback: Callable[..., None], **kwargs) -> Button:
        for k, v in self.__kwargs.items():
            kwargs.setdefault(k, v)
        btn = Button(label=label, **kwargs)
        btn.on_click(callback)
        return btn


class SelectFactory(object):
    def __init__(self, **kwargs):
        self.__kwargs = kwargs

    def __call__(self, label: str,
                 options: list[str],
                 callback: Callable[..., None] | tuple[str, Callable[..., None]] = None,
                 value: str = None,
                 **kwargs) -> Select:
        for k, v in self.__kwargs.items():
            kwargs.setdefault(k, v)

        if value is None or value not in options:
            value = options[0] if len(options) else ""
        ret = Select(title=label, value=value, options=options, **kwargs)

        if callback is not None:
            if isinstance(callback, tuple):
                attr, callback = callback
            else:
                attr = 'value'

            ret.on_change(attr, callback)

        return ret
