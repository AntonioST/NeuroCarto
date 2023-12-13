import abc
from typing import cast, TypeVar

from bokeh.model import Model
from bokeh.models import Div, GlyphRenderer, TextAreaInput
from bokeh.plotting import figure as Figure

__all__ = [
    'UIComponent',
    'RenderComponent',
    'as_layout',
    'col_layout',
    #
    'MessageLogArea',
]

T = TypeVar('T')

class UIComponent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def model(self) -> Model:
        pass


class RenderComponent:
    def plot(self, f: Figure):
        pass

    def set_visible(self, visible: bool, pattern: str = None):
        for name in dir(self):
            render = getattr(self, name)
            if name == 'render' or name.startswith('render_'):
                if pattern is None or pattern in name:
                    if isinstance(render, list):
                        for _render in render:
                            cast(GlyphRenderer, _render).visible = visible
                    elif isinstance(render, dict):
                        for _render in render.values():
                            cast(GlyphRenderer, _render).visible = visible
                    elif isinstance(render, GlyphRenderer):
                        render.visible = visible
            elif isinstance(render, RenderComponent):
                render.set_visible(visible, pattern)

    def list_renders(self, pattern: str = None, recursive: bool = False) -> list[GlyphRenderer]:
        ret = []
        for name in dir(self):
            render = getattr(self, name)
            if name.startswith('render_'):
                if pattern is None or pattern in name:
                    if isinstance(render, list):
                        ret.extend(render)
                    elif isinstance(render, dict):
                        ret.extend(render.values())
                    else:
                        ret.append(render)
            elif isinstance(render, RenderComponent) and recursive:
                ret.extend(render.list_renders(pattern, recursive))
        return ret


def as_layout(model) -> Model:
    return _as_layout(model, 0)


def col_layout(model: list[T], n: int) -> list[list[T]]:
    ret = []
    for i in range(0, len(model), n):
        ret.append(model[i:i + n])
    return ret


def _as_layout(model: list, depth: int) -> Model:
    from bokeh.layouts import column, row

    ret = []
    style = {}
    for component in model:
        if isinstance(component, str):
            ret.append(Div(text=component))
        elif isinstance(component, UIComponent):
            ret.append(component.model())
        elif isinstance(component, list):
            ret.append(_as_layout(component, depth + 1))
        elif isinstance(component, dict):
            style = component
        else:
            ret.append(component)

    if depth % 2 == 0:
        return column(*ret, **style)
    else:
        return row(*ret, **style)


class MessageLogArea(UIComponent):
    def __init__(self, title: str, rows: int, cols: int, **kwargs):
        self.area = TextAreaInput(
            title=title,
            rows=rows,
            cols=cols,
            disabled=True,
            **kwargs
        )

    def model(self) -> Model:
        return self.area

    def log_message(self, *message, reset=False):
        area = self.area

        message = '\n'.join(message)
        area.disabled = False
        try:
            if reset:
                area.value = message
            else:
                text = area.value
                area.value = message + '\n' + text
        finally:
            area.disabled = True

    def clear_message(self):
        area = self.area
        area.disabled = False
        try:
            area.value = ""
        finally:
            area.disabled = True
