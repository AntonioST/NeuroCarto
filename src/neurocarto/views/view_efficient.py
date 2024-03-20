from typing import runtime_checkable, Protocol

from bokeh.models import UIElement, Div

from neurocarto.config import CartoConfig
from neurocarto.probe import ProbeDesp
from neurocarto.util.util_blueprint import BlueprintFunctions
from neurocarto.util.utils import doc_link
from neurocarto.views.base import ViewBase, DynamicView, InvisibleView, ExtensionView

__all__ = ['ElectrodeEfficiencyData', 'ProbeElectrodeEfficiencyProtocol']


@doc_link()
@runtime_checkable
class ProbeElectrodeEfficiencyProtocol(Protocol):
    """
    {ProbeDesp} extension protocol for calculate some statistic values.
    """

    def view_ext_statistics_info(self, bp: BlueprintFunctions) -> dict[str, str]:
        """
        Get some statistics value from a channelmap or a blueprint.

        :param bp:
        :return: dict of {title: value_str}
        """
        pass


class ElectrodeEfficiencyData(ViewBase, ExtensionView, InvisibleView, DynamicView):
    """Display a channel map statistics table."""

    def __init__(self, config: CartoConfig):
        super().__init__(config, logger='neurocarto.view.efficient')

    @property
    def name(self) -> str:
        return 'Channel Efficiency'

    @property
    def description(self) -> str | None:
        return "statistics on channelmap and blueprint"

    @classmethod
    def is_supported(cls, probe: ProbeDesp) -> bool:
        return isinstance(probe, ProbeElectrodeEfficiencyProtocol)

    label_columns_div: Div
    value_columns_div: Div

    def _setup_content(self, **kwargs) -> UIElement:
        from bokeh.layouts import row, column

        self.label_columns_div = Div(text='', visible=False)
        self.value_columns_div = Div(text='', visible=False)

        return row(
            # margin 5 is default
            column(self.label_columns_div, css_classes=['carto-efficient']),
            column(self.value_columns_div, css_classes=['carto-efficient']),
            stylesheets=["""
            div.carto-efficient {
                margin-left: 40px;
                margin-top: 0px;
                margin-bottom: 0px;
            }
            """]
        )

    def on_probe_update(self, probe: ProbeDesp, chmap, electrodes):
        if chmap is not None and isinstance(probe, ProbeElectrodeEfficiencyProtocol):
            # self.logger.debug('on_probe_update()')
            bp = BlueprintFunctions(probe, chmap)
            bp.set_blueprint(electrodes)

            try:
                data = probe.view_ext_statistics_info(bp)
            except BaseException as electrodes:
                self.logger.warning(repr(electrodes), exc_info=electrodes)
                self.label_columns_div.text = ''
                self.value_columns_div.text = ''
            else:
                label = []
                value = []
                for _label, _value in data.items():
                    label.append(f'<div>{_label}</div>')
                    value.append(f'<div>{_value}</div>')
                self.label_columns_div.text = ''.join(label)
                self.value_columns_div.text = ''.join(value)
        else:
            self.label_columns_div.text = ''
            self.value_columns_div.text = ''

        self.label_columns_div.visible = len(self.label_columns_div.text) > 0
        self.value_columns_div.visible = len(self.value_columns_div.text) > 0
