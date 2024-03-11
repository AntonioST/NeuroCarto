from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from chmap.util.util_blueprint import BlueprintFunctions
from chmap.util.utils import SPHINX_BUILD

if TYPE_CHECKING:
    from matplotlib.axes import Axes
elif SPHINX_BUILD:
    ProbePlotElectrodeFunctor = 'chmap.views.blueprint_script.ProbePlotElectrodeFunctor'


def plot_blueprint(bp: BlueprintFunctions,
                   blueprint: NDArray[np.int_],
                   colors: dict[int, Any] = None, *,
                   ax: Axes = None,
                   **kwargs):
    """

    :param bp:
    :param blueprint: Array[category:int, E], where E means all electrodes
    :param colors: categories color {category: color}, where color is used by matplotlib.
    :param ax:  matplotlib.Axes
    :param kwargs:
    :see: {ProbePlotElectrodeFunctor}
    """
    from matplotlib import pyplot as plt
    from chmap.views.blueprint_script import ProbePlotElectrodeFunctor
    from chmap.views.image_plt import RC_FILE

    probe = bp.probe

    if isinstance(functor := probe, ProbePlotElectrodeFunctor) or hasattr(functor, 'view_ext_blueprint_plot_categories'):
        with plt.rc_context(fname=RC_FILE):
            if ax is None:
                fg, ax = plt.subplots()

            functor.view_ext_blueprint_plot_categories(ax, bp.channelmap, blueprint, colors, **kwargs)
    else:
        raise TypeError(f'{type(probe).__name__} not a ProbePlotElectrodeFunctor')
