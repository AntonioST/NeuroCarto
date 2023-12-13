from chmap.probe import ProbeDesp, E, M
from chmap.util.bokeh_view import RenderComponent

__all__ = ['ProbeView']


class ProbeView(RenderComponent):
    def __init__(self, desp: ProbeDesp[M, E]):
        self.desp: ProbeDesp[M, E] = desp
        self.channelmap: M | None = None
        self.electrodes: list[E] | None = None

    def channelmap_desp(self) -> str:
        return self.desp.channelmap_desp(self.channelmap)

    def reset(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0 and self.channelmap is not None:
            channelmap = self.desp.new_channelmap(self.channelmap)
        else:
            channelmap = self.desp.new_channelmap(*args, **kwargs)

        self.channelmap = channelmap
        self.electrodes = self.desp.all_electrodes(channelmap)

    def set_state_for_selected(self, state: int):
        pass

    def set_policy_for_selected(self, state: int):
        pass

    def update(self):
        pass

    def refresh(self):
        pass
