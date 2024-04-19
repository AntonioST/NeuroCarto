import sys
import warnings

from . import select_weaker

__all__ = ['electrode_select']

warnings.warn('this module only used for debugging')

Struct = select_weaker.Struct
electrode_select = select_weaker.electrode_select
pick_electrode = select_weaker.pick_electrode
update_prob = select_weaker.update_prob
category_mapping_probability = select_weaker.category_mapping_probability
information_entropy = select_weaker.information_entropy


def _select_loop(probe_type, s: Struct):
    data = []
    count = 0
    try:
        while (n := s.selected_electrode()) < probe_type.n_channels:
            if (e := pick_electrode(s)) is not None:
                p = s.probability[e]
                update_prob(s, e)
                count += 1
                data.append((n, category_mapping_probability(s.categories[e]), p, information_entropy(s)))
            else:
                break
    except KeyboardInterrupt:
        pass

    import numpy as np
    import matplotlib.pyplot as plt

    with plt.rc_context():
        plt.rcdefaults()

        fg, ax = plt.subplots()
        data = np.array(data)
        data[:, 0] /= probe_type.n_channels
        data[:, 3] /= np.max(data[:, 3])
        ax.plot(data[:, 0], label='Number of selected electrodes')
        ax.plot(data[:, 1], label='Initial probability of selected electrode')
        ax.plot(data[:, 2], label='Actual probability of selected electrode')
        ax.plot(data[:, 3], label='Normalized information entropy')
        ax.set_xlabel('Iteration number')
        ax.legend()


setattr(select_weaker, '_select_loop', _select_loop)


def main(chmap_file: str, debug_fig_file=None, chmap_fig_file=None):
    """
    debug main, run weaker selector and plotting internal state and result.

    :param chmap_file: a channelmap file.
    :param debug_fig_file: output figure name for internal state plotting.
    :param chmap_fig_file: output figure name for selecting outcomes.
    """
    from pathlib import Path
    import matplotlib
    import matplotlib.pyplot as plt
    from neurocarto.probe_npx import NpxProbeDesp, ChannelMap, plot

    file = Path(chmap_file)
    desp = NpxProbeDesp()
    chmap = ChannelMap.from_imro(file)
    blueprint = desp.load_blueprint(file.with_suffix('.blueprint.npy'), desp.all_electrodes(chmap))

    chmap = electrode_select(desp, chmap, blueprint)
    if debug_fig_file is None:
        plt.show()
    else:
        plt.savefig(Path(debug_fig_file).with_suffix('.png'))

    if chmap_fig_file is not None:
        with plt.rc_context():
            rc = matplotlib.rc_params_from_file('tests/default.matplotlibrc', fail_on_error=True, use_default_template=True)
            plt.rcParams.update(rc)

            fg, ax = plt.subplots()
            height = 6
            plot.plot_channelmap_block(ax, chmap, height=height, color='k', shank_width_scale=2)
            plot.plot_probe_shape(ax, chmap, height=height, color='gray', label_axis=True, shank_width_scale=2)

            plt.savefig(Path(chmap_fig_file).with_suffix('.png'))


if __name__ == '__main__':
    main(*sys.argv[1:])
