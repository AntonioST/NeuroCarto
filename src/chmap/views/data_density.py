from __future__ import annotations

from chmap.config import ChannelMapEditorConfig
from chmap.views.data import Data1DView

__all__ = ['ElectrodeDensityDataView']


class ElectrodeDensityDataView(Data1DView):
    """Show electrode (selected) density curve beside the shank."""

    def __init__(self, config: ChannelMapEditorConfig):
        super().__init__(config, logger='chmap.view.density')

    @property
    def name(self) -> str:
        return 'Electrode Density Curve'

    @property
    def description(self) -> str | None:
        return 'show electrode density curve along the shanks'

    def data(self):
        from chmap.probe_npx.npx import ChannelMap

        if isinstance(self.channelmap, ChannelMap):
            from chmap.probe_npx.stat import npx_electrode_density
            try:
                return self.arr_to_dict(self.transform(npx_electrode_density(self.channelmap), vmax=1))
            except RuntimeError as e:
                self.logger.warning('update density data fail', exc_info=e)

        return None
