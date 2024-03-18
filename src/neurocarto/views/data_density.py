from __future__ import annotations

from typing import runtime_checkable, Protocol, Any

import numpy as np
from numpy.typing import NDArray

from neurocarto.config import CartoConfig
from neurocarto.util.utils import doc_link, SPHINX_BUILD
from neurocarto.views import ExtensionView
from neurocarto.views.data import Data1DView

if SPHINX_BUILD:
    ProbeDesp = 'neurocarto.probe.ProbeDesp'

__all__ = [
    'ElectrodeDensityDataView',
    'ProbeElectrodeDensityProtocol'
]


@doc_link()
@runtime_checkable
class ProbeElectrodeDensityProtocol(Protocol):
    """
    {ProbeDesp} extension protocol for calculate electrode density distribution curve.
    """

    def view_ext_electrode_density(self, chmap: Any) -> NDArray[np.float_]:
        """
        Calculate electrode density along the probe.

        :param chmap:
        :return: Array[float, [S,], (v, y), Y] density array
        """
        pass


@doc_link()
class ElectrodeDensityDataView(Data1DView, ExtensionView):
    """
    Show electrode (selected) density curve beside the shank.

    Check whether the {ProbeDesp} implement protocol {ProbeElectrodeDensityProtocol}.
    """

    def __init__(self, config: CartoConfig):
        super().__init__(config, logger='neurocarto.view.density')

    @property
    def name(self) -> str:
        return 'Electrode Density Curve'

    @property
    def description(self) -> str | None:
        return 'show electrode density curve along the shanks'

    @classmethod
    def is_supported(cls, probe: ProbeDesp) -> bool:
        return isinstance(probe, ProbeElectrodeDensityProtocol)

    def data(self):
        if (chmap := self.channelmap) is not None and isinstance((functor := self.probe), ProbeElectrodeDensityProtocol):
            try:
                return self.arr_to_dict(self.transform(functor.view_ext_electrode_density(chmap), vmax=1))
            except RuntimeError as e:
                self.logger.warning('update density data fail', exc_info=e)

        return None
