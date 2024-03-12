from typing import Protocol

from neurocarto.util.utils import import_name, doc_link
from .desp import NpxProbeDesp, NpxElectrodeDesp
from .npx import ChannelMap

__all__ = ['electrode_select', 'load_select', 'ElectrodeSelector']

BUILTIN_SELECTOR = {
    'default': 'neurocarto.probe_npx.select_default:electrode_select',
    'weaker': 'neurocarto.probe_npx.select_weaker:electrode_select',
}


class ElectrodeSelector(Protocol):
    """An electrode selector protocol class."""

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


@doc_link()
def load_select(selector: str) -> ElectrodeSelector:
    """
    Load a Neuropixels electrode selector.

    :param selector: ``module_path:name``
    :return:
    :see: {import_name()}
    """
    selector = BUILTIN_SELECTOR.get(selector, selector)
    return import_name('selector', selector)


@doc_link()
def electrode_select(desp: NpxProbeDesp, chmap: ChannelMap, blueprint: list[NpxElectrodeDesp], *,
                     selector: str = 'default',
                     **kwargs) -> ChannelMap:
    """
    Do electrode selection for the Neuropixels channel map with a given blueprint.

    :param desp:
    :param chmap: channelmap type. It is a reference.
    :param blueprint: channelmap blueprint
    :param selector: selector name
    :param kwargs: selector keyword parameters.
    :return: {load_select()}
    :see:
    """
    selector = load_select(selector)

    return selector(desp, chmap, blueprint, **kwargs)
