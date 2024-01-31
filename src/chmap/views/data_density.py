from __future__ import annotations

from chmap.config import ChannelMapEditorConfig
from chmap.probe import ProbeDesp
from chmap.probe_npx import NpxProbeDesp
from chmap.util.bokeh_app import run_later
from chmap.views.data import Data1DView

__all__ = ['ElectrodeDensityDataView']


class ElectrodeDensityDataView(Data1DView):
    """Show electrode (selected) density curve beside the shank."""

    def __init__(self, config: ChannelMapEditorConfig):
        super().__init__(config, logger='chmap.view.density')

        self._data = None

    @property
    def name(self) -> str:
        return 'Electrode Density Curve'

    @property
    def description(self) -> str | None:
        return 'show electrode density curve along the shanks'

    def on_probe_update(self, probe: ProbeDesp, chmap, e):
        if chmap is None:
            self._data = None
        elif isinstance(probe, NpxProbeDesp):
            # self.logger.debug('on_probe_update()')

            try:
                from chmap.probe_npx.stat import npx_electrode_density
                self._data = self.arr_to_dict(npx_electrode_density(probe, chmap))
            except RuntimeError as ex:
                self.logger.warning(repr(ex), exc_info=ex)
                self._data = None

        run_later(self.update)

    def data(self):
        return self._data
