import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

from neurocarto.probe_npx import ChannelMap, plot
from neurocarto.util.util_blueprint import BlueprintFunctions

APPEND_MODE = ('overwrite', 'min', 'mean', 'max')

AP = argparse.ArgumentParser(description="""\
Read SpikeGlx files and calculate the AP rms.

Corresponding to the Fig.5c in paper.
""")
AP.add_argument('-o', '--output', metavar='FILE', type=Path, required=True, dest='OUTPUT',
                help='output .npy file.')
AP.add_argument('-a', '--append', nargs='?', choices=APPEND_MODE, default=None, const='overwrite', dest='append',
                help='append result into output file.')
AP.add_argument('-s', '--save', nargs='?', const='', default=None, dest='save_figure',
                help='save figure. use -o when the value is omitted.')
AP.add_argument('-f', '--force', action='store_true', dest='force',
                help='force re-generate output .npy file')
AP.add_argument('--ap', metavar='LO,HI', default='1000,5000', dest='ap_band',
                help='high pass filter band. (default=1000,5000)')
AP.add_argument('--car', choices=('none', 'global', 'local'), default='global', dest='car',
                help='common artifact removing along channels.')
AP.add_argument('--sample-times', metavar='NUM', type=int, default=10, dest='sample_times',
                help='number of sample segments. (default=10)')
AP.add_argument('--sample-duration', metavar='SEC', type=float, default=1, dest='sample_duration',
                help='duration of the sample segments. (default=1)')
AP.add_argument(metavar='GLX_FILES', nargs='+', type=Path, action='extend', dest='FILES',
                help='Spike Glx files.')

# local_car_window =100
local_car_window = (100, 300)  # um


def uniform_sample(r, d: float, t: int):
    """
    uniform random sample non-overlap segments.

    :param r: range
    :param d: duration
    :param t: sample times
    :return: [(start, stop)], not overlapped
    """
    a = r[1] - r[0]
    if a < d * t:
        raise ValueError(f'd(={d}) * t(={t}) larger than r(={r})')

    import itertools

    b = np.random.random(t + 1)
    b *= (a - d * t) / np.sum(b)
    b = itertools.accumulate(b[:t], func=lambda ac, it: ac + it + d)
    return [(c + r[0], c + r[0] + d) for c in b]


def open_glx(glx_file: Path, glx_map: ChannelMap) -> np.ndarray:
    """
    open glx binary file as a numpy memmap file.

    Adapt from readSGLX.makeMemMapRaw.

    :param glx_file:
    :param glx_map:
    :return:
    """
    try:
        n_channel = int(glx_map.meta['nSavedChans'])
    except (KeyError, ValueError):
        n_channel = glx_map.n_channels + 1  # 1 sync channel

    try:
        size = int(glx_map.meta['fileSizeBytes'])
    except (KeyError, ValueError):
        size = glx_file.stat().st_size

    samples = size // (2 * n_channel)
    return np.memmap(glx_file, dtype='int16', mode='r', shape=(n_channel, samples), offset=0, order='F')


def construct_filter(ap_band: tuple[float, float], fs: float):
    from scipy.signal import firwin, kaiserord
    bandwidth = ap_band[1] - ap_band[0]
    width = bandwidth / 4
    n, beta = kaiserord(60, width * 2 / fs)
    n += (n + 1) % 2  # as odd number
    return firwin(n, ap_band, window=("kaiser", beta), pass_zero='bandpass', scale=False, nyq=fs / 2)


def apply_filter(data: np.ndarray, f: np.ndarray):
    from scipy.signal import convolve
    f = np.expand_dims(f, axis=0)

    n = (len(f) - 1) // 2
    pw = np.zeros((data.ndim, 2), dtype=int)
    pw[-1] = (n, n)

    data = np.pad(data, pw, mode='reflect', reflect_type='odd')
    return convolve(data, f, mode='valid')


def build_local_car_operator(pos: np.ndarray, window: float | tuple[float, float]) -> np.ndarray:
    """

    :param pos: Array[pos_um:int, C, 2]
    :param window: a radius, or a radius range
    :return: Array[float, C, C]
    """
    x = pos[:, 0]  # Array[int, C]
    y = pos[:, 1]
    dx = np.abs(np.subtract.outer(x, x))  # Array[int, C, C]
    dy = np.abs(np.subtract.outer(y, y))
    dd = np.sqrt(dx ** 2 + dy ** 2)  # Array[um:float, C, C]

    n = len(pos)
    ret = np.zeros_like(dd)

    match window:
        case int(radius) | float(radius):
            for i in range(n):
                c = dd[i] <= radius
                if np.any(c):
                    ret[i, c] = -1 / np.count_nonzero(c)
                ret[i, i] = 1
        case (int(exclude) | float(exclude), int(radius) | float(radius)):
            for i in range(n):
                c = (exclude <= dd[i]) & (dd[i] <= radius)
                if np.any(c):
                    ret[i, c] = -1 / np.count_nonzero(c)
                ret[i, i] = 1
        case _:
            raise TypeError()

    return ret


def segment_glx_data(glx_file: Path, glx_map: ChannelMap,
                     sample_times=10,
                     sample_duration=1,
                     ap_band: tuple[int, int] = (1000, 5000),
                     car='global'):
    glx_data = open_glx(glx_file, glx_map)

    total_channel, total_time_step = glx_data.shape
    sample_rate = int(glx_map.meta['imSampRate'])
    total_duration = total_time_step / sample_rate

    sample_time_range = uniform_sample((0, total_duration), sample_duration, sample_times)

    f = construct_filter(ap_band, sample_rate)
    if car == 'local':
        c = build_local_car_operator(glx_map.channel_pos, local_car_window)

    out = np.zeros((total_channel - 1,), dtype=float)  # Array[float, C]
    for i, (t0, t1) in enumerate(sample_time_range):
        print(f'process {i + 1}/{len(sample_time_range)} ...', end='\r')

        t = slice(int(t0 * sample_rate), int(t1 * sample_rate))

        # excluding sync channel
        data = glx_data[:-1, t]  # Array[int, C, T]

        # as voltage
        data = data.astype(float) * 0.195  # Array[float, C, T]

        # high pass
        data = apply_filter(data, f)  # Array[float, C, T]

        if car == 'global':
            # median car on C
            data -= np.median(data, axis=0)  # Array[float, C, T]
        elif car == 'local':
            # Array[float, C, C] @ Array[float, C, T]
            data = c @ data  # Array[float, C, T]

        data = np.sqrt(np.mean(np.power(data, 2), axis=1))  # Array[float, C]
        np.add(out, data, out=out)

    print(' ' * 80, end='\r')
    return out / len(sample_time_range)  # Array[float, C]


def compute_data(bp: BlueprintFunctions, opt) -> np.ndarray:
    ap_lo, _, ap_hi = str(opt.ap_band).partition(',')
    ap_band = int(ap_lo), int(ap_hi)
    print('AP band', ap_band)

    sample_times = opt.sample_times
    sample_duration = opt.sample_duration
    print('sample', sample_duration, 'second', sample_times, 'times')

    print('load meta ...')
    glx_files = opt.FILES

    append_mode: str = opt.append

    output_file: Path = opt.OUTPUT
    if append_mode is None or not output_file.exists():
        output_file.unlink(missing_ok=True)
        ret = np.full((len(bp),), np.nan, dtype=float)
    else:
        assert output_file.exists()
        assert append_mode in APPEND_MODE
        print(f'read {output_file}')
        ret = np.load(output_file)

    if append_mode in ('min', 'mean', 'max'):
        ret = [ret]

    for i, glx_file in enumerate(glx_files):
        glx_file = glx_file.with_suffix('.bin')
        glx_map = ChannelMap.from_meta(glx_file.with_suffix('.meta'))

        print(f'load [{i + 1}/{len(glx_files)}] {glx_file} ...')
        glx_rms = segment_glx_data(glx_file, glx_map, sample_times, sample_duration, ap_band, opt.car)

        if append_mode not in ('min', 'mean', 'max'):
            bp.put_data(ret, glx_map, glx_rms)
        else:
            tmp = np.full_like(ret[0], np.nan)
            bp.put_data(tmp, glx_map, glx_rms)
            ret.append(tmp)

    match append_mode:
        case 'min':
            ret = np.nanmin(ret, axis=0)
        case 'max':
            ret = np.nanmax(ret, axis=0)
        case 'mean':
            ret = np.nanmean(ret, axis=0)

    print(f'save {output_file}')
    np.save(output_file, ret)

    return ret


def plot_data(bp: BlueprintFunctions, data: np.ndarray, opt):
    with plt.rc_context(fname='tests/default.matplotlibrc'):
        fig, ax = plt.subplots(gridspec_kw=dict(top=0.90))

        plot_data_matrix(bp, ax, data)
        # plot_data_scatter(bp, ax, data)
        # plot_data_curve(bp, ax, data)

        ax.set_ylim(0, 6)

        if opt.save_figure is None:
            print('show...')
            plt.show()
        else:
            save_figure = opt.save_figure
            if len(save_figure) == 0:
                save_figure = Path(opt.OUTPUT).with_suffix('.png')

            print(f'save {save_figure}')
            plt.savefig(save_figure, dpi=600)


def plot_data_matrix(bp: BlueprintFunctions, ax: Axes, data: np.ndarray):
    im = plot.plot_electrode_matrix(
        ax, bp.channelmap.probe_type, data, 'raw',
        shank_list=[3, 2, 1, 0],
        kernel=(0, 1),  # (C, R)
        vmax=6,
        vmin=0,
    )

    insert_colorbar(ax, im)


def plot_data_scatter(bp: BlueprintFunctions, ax: Axes, data: np.ndarray):
    i = np.nonzero(~np.isnan(data))[0]
    s = np.array([3, 2, 1, 0])
    data -= np.nanmin(data)
    x = s[bp.s[i]] + data[i] / np.nanmax(data)  # shank as x
    y = bp.y[i] / 1000  # mm
    ax.scatter(2 * x, y, s=1, c='g')

    for x in np.unique(bp.s):
        ax.axvline(2 * x, color='gray', lw=0.5)


def plot_data_curve(bp: BlueprintFunctions, ax: Axes, data: np.ndarray):
    lines, y = plot.cast_electrode_curve(bp.channelmap.probe_type, data, 'raw', kernel='norm')

    # normalize
    lines -= np.min(lines)
    lines /= np.max(lines) / 2

    y = y / 1000  # mm

    shank_list = np.array([3, 2, 1, 0])
    for i, s in enumerate(shank_list):
        ax.plot(lines[s] + 2 * i, y, color='k', )


def insert_colorbar(ax: Axes, im):
    cax = ax.inset_axes([0, 1.1, 1, 0.02])  # [x0, y0, width, height]
    return ax.figure.colorbar(im, cax=cax, orientation='horizontal')


def main(opt):
    print('init ...')
    bp = BlueprintFunctions('npx', 24)

    output_file: Path = opt.OUTPUT
    if opt.force or not output_file.exists():
        data = compute_data(bp, opt)
    else:
        data = np.load(output_file)

    plot_data(bp, data, opt)


if __name__ == '__main__':
    main(AP.parse_args())
