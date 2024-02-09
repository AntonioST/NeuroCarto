import warnings

from . import select_weaker

__all__ = ['electrode_select']

warnings.warn('this module only used for debugging')

electrode_select = select_weaker.electrode_select
selected_electrode = select_weaker.selected_electrode
pick_electrode = select_weaker.pick_electrode
update_prob = select_weaker.update_prob
category_mapping_probability = select_weaker.category_mapping_probability
information_entropy = select_weaker.information_entropy


def _select_loop(desp, probe_type, cand):
    data = []
    count = 0
    try:
        while (n := selected_electrode(cand)) < probe_type.n_channels:
            if (e := pick_electrode(cand)) is not None:
                p = e.prob
                update_prob(desp, cand, e)
                count += 1
                data.append((n, category_mapping_probability(e.category), p, information_entropy(cand)))
            else:
                break
    except KeyboardInterrupt:
        pass

    import numpy as np
    import matplotlib.pyplot as plt

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
