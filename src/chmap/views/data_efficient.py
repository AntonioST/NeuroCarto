import logging
from typing import NamedTuple

from bokeh.models import UIElement, Div

from chmap.config import ChannelMapEditorConfig
from chmap.probe import ProbeDesp
from chmap.probe_npx import NpxProbeDesp, ChannelMap, NpxElectrodeDesp
from chmap.views.base import ViewBase, DynamicView, InvisibleView

__all__ = ['ElectrodeEfficiencyData']


class ElectrodeEfficientStat(NamedTuple):
    total_channel: int
    used_channel: int
    used_channel_on_shanks: list[int]

    require_electrodes: float
    channel_efficiency: float
    remain_channel: int  # number of electrode selected in remainder policy
    remain_electrode: int  # number of electrode set in remainder policy


def make_stat_div(text: str):
    return Div(text=text, margin=2)


class ElectrodeEfficiencyData(ViewBase, InvisibleView, DynamicView):
    label_used_channel: Div = make_stat_div('used channels')
    label_require_electrodes: Div = make_stat_div('require electrodes')
    label_channel_efficiency: Div = make_stat_div('channel efficiency')
    label_remain_channel: Div = make_stat_div('remain channels')
    label_remain_electrode: Div = make_stat_div('remain electrode')

    def __init__(self, config: ChannelMapEditorConfig):
        self.logger = logging.getLogger('chmap.view.efficient')
        self.logger.debug('init()')

        super().__init__(config)

        self._stat: ElectrodeEfficientStat | None = None

    @property
    def name(self) -> str:
        return 'Electrode Efficiency'

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

    def on_probe_update(self, probe: ProbeDesp, chmap, e):
        if chmap is None:
            self._stat = None
        elif isinstance(probe, NpxProbeDesp):
            # self.logger.debug('on_probe_update()')

            try:
                self._stat = electrode_efficient_npx(probe, chmap, e)
            except BaseException as e:
                self.logger.warning(repr(e), exc_info=e)
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

            self.label_require_electrodes.text = f'{stat.require_electrodes}'
            self.label_channel_efficiency.text = f'{100 * stat.channel_efficiency:.2f}%'
            self.label_remain_channel.text = f'{stat.remain_channel}'
            self.label_remain_electrode.text = f'{stat.remain_electrode}'


def electrode_efficient_npx(probe: NpxProbeDesp, chmap: ChannelMap, e: list[NpxElectrodeDesp]) -> ElectrodeEfficientStat:
    used_channel = len(chmap)
    used_channel_on_shanks = [
        len([it for it in chmap.electrodes if it.shank == s])
        for s in range(chmap.probe_type.n_shank)
    ]

    p, c = _electrode_efficient_npx_require_electrodes(e)
    cp = 0 if p == 0 else c / p
    re, rc = _get_electrode(e, [NpxProbeDesp.POLICY_REMAINDER, NpxProbeDesp.POLICY_UNSET])

    return ElectrodeEfficientStat(
        chmap.probe_type.n_channels,
        used_channel,
        used_channel_on_shanks,
        require_electrodes=p,
        channel_efficiency=cp,
        remain_electrode=re,
        remain_channel=rc
    )


def _electrode_efficient_npx_require_electrodes(e: list[NpxElectrodeDesp]) -> tuple[float, int]:
    p0, s0 = _get_electrode(e, [NpxProbeDesp.POLICY_SET, NpxProbeDesp.POLICY_D1])
    p2, s2 = _get_electrode(e, [NpxProbeDesp.POLICY_D2])
    p4, s4 = _get_electrode(e, [NpxProbeDesp.POLICY_D4])
    p = p0 + p2 / 2 + p4 / 4
    s = s0 + s2 + s4
    return p, s


def _get_electrode(e: list[NpxElectrodeDesp], policies: list[int]) -> tuple[int, int]:
    e1 = [it for it in e if it.policy in policies]
    e2 = [it for it in e1 if it.state == NpxProbeDesp.STATE_USED]
    return len(e1), len(e2)
