from chmap.probe import ProbeDesp
from chmap.util.bokeh_view import RenderComponent

__all__ = ['ProbeView']


class ProbeView(RenderComponent):
    def __init__(self, desp: ProbeDesp):
        self._desp: ProbeDesp = desp
