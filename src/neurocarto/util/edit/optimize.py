import numpy as np
from numpy.typing import NDArray

from neurocarto.probe_npx.npx import ChannelMap
from neurocarto.probe_npx.stat import npx_channel_efficiency
from neurocarto.util.util_blueprint import BlueprintFunctions

__all__ = ['optimize_channelmap', 'generate_channelmap']


def optimize_channelmap(bp: BlueprintFunctions,
                        chmap: ChannelMap,
                        blueprint: NDArray[int],
                        sample_times: int = 100,
                        **kwargs) -> tuple[ChannelMap, float, float]:
    """
    Sample and find the optimized channelmap that has maxima channel efficiency.

    :param bp:
    :param chmap: initial channel map
    :param blueprint:
    :param sample_times: (int=100)
    :param kwargs: selector parameters
    :return: tuple of (channelmap, aeff, ceff)
    """
    blueprint_lst = bp.apply_blueprint(blueprint=blueprint)

    max_map = chmap
    max_aef, max_cef = npx_channel_efficiency(bp, chmap, blueprint)

    for i in range(sample_times):
        chmap = bp.select_electrodes(chmap, blueprint_lst, **kwargs)
        aeff, ceff = npx_channel_efficiency(bp, chmap, blueprint)
        if ceff > max_cef:
            max_map = chmap
            max_aef = aeff
            max_cef = ceff

    return max_map, max_aef, max_cef


def generate_channelmap(bp: BlueprintFunctions,
                        chmap: ChannelMap,
                        blueprint: NDArray[int],
                        sample_times: int = 100,
                        **kwargs) -> tuple[list[ChannelMap], NDArray[float], NDArray[float]]:
    """
    generate a group of channel maps.

    :param bp:
    :param chmap: channelmap type. It is a reference.
    :param blueprint:
    :param sample_times: N (int=100)
    :param kwargs: selector parameters
    :return: tuple of ([channelmap], Array[aeff, N], Array[ceff, N])
    """
    bp = bp.clone(pure=True)

    blueprint_lst = bp.apply_blueprint(blueprint=blueprint)

    ret_map = []
    ret_aef = []
    ret_cef = []

    for i in range(sample_times):
        chmap = bp.select_electrodes(chmap, blueprint_lst, **kwargs)
        aeff, ceff = npx_channel_efficiency(bp, chmap, blueprint)
        ret_map.append(chmap)
        ret_aef.append(aeff)
        ret_cef.append(ceff)

    return ret_map, np.array(ret_aef), np.array(ret_cef)
