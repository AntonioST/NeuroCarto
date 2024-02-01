from typing import Protocol

from chmap.util.utils import import_func
from .desp import NpxProbeDesp, NpxElectrodeDesp
from .npx import ChannelMap

__all__ = ['electrode_select']

BUILTIN_SELECTOR = {
    'default': 'chmap.probe_npx.select_default:electrode_select',
    'weaker': 'chmap.probe_npx.select_weaker:electrode_select',
}


class ElectrodeSelector(Protocol):
    def __call__(self, desp: NpxProbeDesp, chmap: ChannelMap, s: list[NpxElectrodeDesp], **kwargs) -> ChannelMap:
        pass


def load_select(selector: str) -> ElectrodeSelector:
    return import_func('selector', selector)


def electrode_select(desp: NpxProbeDesp, chmap: ChannelMap, s: list[NpxElectrodeDesp], *,
                     selector: str = 'default',
                     **kwargs) -> ChannelMap:
    selector = BUILTIN_SELECTOR.get(selector, selector)
    selector = load_select(selector)

    return selector(desp, chmap, s, **kwargs)
