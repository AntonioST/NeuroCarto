from typing import Protocol

from chmap.util.utils import import_name
from .desp import NpxProbeDesp, NpxElectrodeDesp
from .npx import ChannelMap

__all__ = ['electrode_select', 'load_select']

BUILTIN_SELECTOR = {
    'default': 'chmap.probe_npx.select_default:electrode_select',
    'weaker': 'chmap.probe_npx.select_weaker:electrode_select',
}


class ElectrodeSelector(Protocol):
    def __call__(self, desp: NpxProbeDesp, chmap: ChannelMap, blueprint: list[NpxElectrodeDesp], **kwargs) -> ChannelMap:
        """
        Selecting electrodes based on the electrode blueprint.

        :param desp:
        :param chmap: channelmap type. It is a reference.
        :param blueprint: channelmap blueprint
        :param kwargs: other parameters.
        :return: generated channelmap
        """
        pass


def load_select(selector: str) -> ElectrodeSelector:
    selector = BUILTIN_SELECTOR.get(selector, selector)
    return import_name('selector', selector)


def electrode_select(desp: NpxProbeDesp, chmap: ChannelMap, blueprint: list[NpxElectrodeDesp], *,
                     selector: str = 'default',
                     **kwargs) -> ChannelMap:
    selector = load_select(selector)

    return selector(desp, chmap, blueprint, **kwargs)
