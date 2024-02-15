from typing import TYPE_CHECKING

from bokeh.models import UIElement, Div

from chmap.config import ChannelMapEditorConfig
from chmap.probe import ProbeDesp
from chmap.probe_npx import NpxProbeDesp
from chmap.views.base import ViewBase, DynamicView, InvisibleView

if TYPE_CHECKING:
    from chmap.probe_npx.stat import ElectrodeEfficiencyStat

__all__ = ['ElectrodeEfficiencyData']


def make_stat_div(text: str):
    return Div(text=text, margin=2)


class ElectrodeEfficiencyData(ViewBase, InvisibleView, DynamicView):
    """Display a channel map statistics table."""

    label_used_channel: Div = make_stat_div('used channels')
    label_request_electrodes: Div = make_stat_div('request electrodes')
    label_channel_efficiency: Div = make_stat_div('channel efficiency')
    label_remain_channel: Div = make_stat_div('remain channels')
    label_remain_electrode: Div = make_stat_div('remain electrode')

    def __init__(self, config: ChannelMapEditorConfig):
        super().__init__(config, logger='chmap.view.efficient')

        self._stat: ElectrodeEfficiencyStat | None = None

    @property
    def name(self) -> str:
        return 'Channel Efficiency'

    @property
    def description(self) -> str | None:
        return "statistics on channelmap and blueprint"

    _label_columns: list[Div]
    _value_columns: list[Div]

    def _setup_content(self, **kwargs) -> UIElement:
        from bokeh.layouts import row, column

        self._label_columns = []
        self._value_columns = []

        for attr in ElectrodeEfficiencyData.__annotations__:
            if attr.startswith('label_') and (div := getattr(ElectrodeEfficiencyData, attr, None), Div):
                self._label_columns.append(make_stat_div(div.text))  # copy, avoid reused in another document.
                self._value_columns.append(value := make_stat_div(''))
                setattr(self, attr, value)

        return row(
            # margin 5 is default
            column(self._label_columns, margin=(5, 5, 5, 5)),
            column(self._value_columns, margin=(5, 5, 5, 20)),
            margin=(5, 5, 5, 40)
        )

    def on_probe_update(self, probe: ProbeDesp, chmap, electrodes):
        if chmap is None:
            self._stat = None
        elif isinstance(probe, NpxProbeDesp):
            # self.logger.debug('on_probe_update()')

            try:
                from chmap.probe_npx.stat import npx_channel_efficiency
                self._stat = npx_channel_efficiency(chmap, electrodes)
            except BaseException as electrodes:
                self.logger.warning(repr(electrodes), exc_info=electrodes)
                self._stat = None

        self.update()

    def start(self):
        self.update()

    def update(self):
        if (stat := self._stat) is None:
            for label in self._value_columns:
                label.text = ''
        else:
            ucs = ', '.join(map(lambda it: f's{it[0]}={it[1]}', enumerate(stat.used_channel_on_shanks)))
            self.label_used_channel.text = f'{stat.used_channel}, total={stat.total_channel}, ({ucs})'

            self.label_request_electrodes.text = f'{stat.request_electrodes}'
            self.label_channel_efficiency.text = f'{100 * stat.channel_efficiency:.2f}%'
            self.label_remain_channel.text = f'{stat.remain_channel}'
            self.label_remain_electrode.text = f'{stat.remain_electrode}'
