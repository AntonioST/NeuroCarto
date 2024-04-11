from __future__ import annotations

import textwrap
from typing import Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from neurocarto.util.util_blueprint import BlueprintFunctions
from neurocarto.util.utils import SPHINX_BUILD, doc_link

if TYPE_CHECKING:
    from matplotlib.axes import Axes
elif SPHINX_BUILD:
    ProbePlotElectrodeProtocol = 'neurocarto.views.blueprint_script.ProbePlotElectrodeProtocol'

__all__ = ['plot_blueprint']


@doc_link(DOC=textwrap.dedent(BlueprintFunctions.plot_blueprint.__doc__))
def plot_blueprint(bp: BlueprintFunctions,
                   blueprint: NDArray[np.int_],
                   colors: dict[int, Any] = None, *,
                   ax: Axes = None,
                   **kwargs):
    """
    {DOC}
    :see: {BlueprintFunctions#plot_blueprint()}
    """
    from matplotlib import pyplot as plt
    from neurocarto.views.blueprint_script import ProbePlotElectrodeProtocol
    from neurocarto.views.image_plt import RC_FILE

    probe = bp.probe

    if isinstance(functor := probe, ProbePlotElectrodeProtocol) or hasattr(functor, 'view_ext_blueprint_plot_categories'):
        with plt.rc_context(fname=RC_FILE):
            if ax is None:
                fg, ax = plt.subplots()

            functor.view_ext_blueprint_plot_categories(ax, bp.channelmap, blueprint, colors, **kwargs)
    else:
        raise TypeError(f'{type(probe).__name__} not a ProbePlotElectrodeProtocol')
