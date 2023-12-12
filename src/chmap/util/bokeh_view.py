import abc
from typing import cast

from bokeh.model import Model
from bokeh.models import Div, GlyphRenderer
from bokeh.plotting import figure as Figure

__all__ = [
    'UIComponent',
    'as_layout'
]


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
