import numpy as np
from numpy.typing import NDArray

from neurocarto.probe_npx.npx import ChannelMap
from neurocarto.probe_npx.stat import npx_channel_efficiency
from neurocarto.util.util_blueprint import BlueprintFunctions

__all__ = ['optimize_channelmap', 'generate_channelmap']


def optimize_channelmap(bp: BlueprintFunctions,
                        chmap: ChannelMap,
                        blueprint: NDArray[int],
                        sample_times: int = 100, *,
                        n_worker: int = 1,
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
    bp = bp.clone(pure=True)

    if n_worker < 0:
        raise ValueError()
    elif n_worker in (0, 1):
        return _optimize_channelmap(bp, chmap, blueprint, sample_times, **kwargs)
    else:
        import multiprocessing
        with multiprocessing.Pool(n_worker) as pool:

            _sample_times = sample_times // n_worker
            sample_times_list = [_sample_times] * n_worker
            sample_times_list[-1] += sample_times - sum(sample_times_list)

            jobs = [
                pool.apply_async(_optimize_channelmap, (bp, chmap, blueprint, t), kwargs)
                for t in sample_times_list
            ]

            pool.close()
            pool.join()

            max_map = None
            max_aef = None
            max_cef = -1

            for i, job in enumerate(jobs):
                chmap, aeff, ceff = job.get()
                print(i, ceff)
                if ceff > max_cef:
                    max_map = chmap
                    max_aef = aeff
                    max_cef = ceff

            pool.terminate()

        return max_map, max_aef, max_cef


def _optimize_channelmap(bp: BlueprintFunctions,
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
                        sample_times: int = 100, *,
                        n_worker: int = 1,
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

    if n_worker < 0:
        raise ValueError()
    elif n_worker in (0, 1):
        return _generate_channelmap(bp, chmap, blueprint, sample_times, **kwargs)
    else:
        import multiprocessing
        with multiprocessing.Pool(n_worker) as pool:

            _sample_times = sample_times // n_worker
            sample_times_list = [_sample_times] * n_worker
            sample_times_list[-1] += sample_times - sum(sample_times_list)

            jobs = [
                pool.apply_async(_generate_channelmap, (bp, chmap, blueprint, t), kwargs)
                for t in sample_times_list
            ]

            ret_map = []
            ret_aef = []
            ret_cef = []
            for job in jobs:
                chmap, aeff, ceff = job.get()
                ret_map.extend(chmap)
                ret_aef.append(aeff)
                ret_cef.append(ceff)

        return ret_map, np.concatenate(ret_aef), np.concatenate(ret_cef)


def _generate_channelmap(bp: BlueprintFunctions,
                         chmap: ChannelMap,
                         blueprint: NDArray[int],
                         sample_times: int = 100,
                         **kwargs) -> tuple[list[ChannelMap], NDArray[float], NDArray[float]]:
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
