import logging
from typing import NamedTuple

from bokeh.models import UIElement, Div
from bokeh.plotting import figure as Figure

from chmap.config import ChannelMapEditorConfig
from chmap.probe import ProbeDesp
from chmap.probe_npx import NpxProbeDesp, ChannelMap, NpxElectrodeDesp
from chmap.views.base import ViewBase, DynamicView, InvisibleView

__all__ = ['ElectrodeEfficientData']


class ElectrodeEfficientStat(NamedTuple):
    total_channel: int
    used_channel: int
    used_channel_on_shanks: list[int]

    require_channel: float
    remain_channel: int  # number of electrode selected in remainder policy
    remain_electrode: int  # number of electrode set in remainder policy


def make_stat_div(text: str):
    return Div(text=text, margin=2)


class ElectrodeEfficientData(ViewBase, InvisibleView, DynamicView):
    label_used_channel: Div = make_stat_div('used channels')
    label_require_channel: Div = make_stat_div('require channels')
    label_remain_channel: Div = make_stat_div('remain channels')
    label_remain_electrode: Div = make_stat_div('remain electrode')

    def __init__(self, config: ChannelMapEditorConfig):
        self.logger = logging.getLogger('chmap.view.efficient')
        self.logger.debug('init()')

        super().__init__(config)

        self._stat: ElectrodeEfficientStat | None = None

    @property
    def name(self) -> str:
        return 'Electrode Efficient'

    _label_columns: list[Div]
    _value_columns: list[Div]
    _content: UIElement

    def setup(self, f: Figure, **kwargs) -> list[UIElement]:
        self.logger.debug('setup()')

        from bokeh.layouts import row, column

        self._label_columns = []
        self._value_columns = []

        for attr in ElectrodeEfficientData.__annotations__:
            if attr.startswith('label_') and (div := getattr(ElectrodeEfficientData, attr, None), Div):
                self._label_columns.append(div)
                self._value_columns.append(value := make_stat_div(''))
                setattr(self, attr, value)

        self._content = row(
            # margin 5 is default
            column(self._label_columns, margin=(5, 5, 5, 5)),
            column(self._value_columns, margin=(5, 5, 5, 20)),
            margin=(5, 5, 5, 40)
        )

        return [
            row(self.setup_visible_switch(), Div(text=f'<b>{self.name}</b>')),
            self._content
        ]

    def on_visible(self, visible: bool):
        self._content.visible = visible

    def on_probe_update(self, probe: ProbeDesp, chmap, e):
        if chmap is None:
            self._stat = None
        elif isinstance(probe, NpxProbeDesp):
            self._stat = electrode_efficient_npx(probe, chmap, e)

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

            self.label_require_channel.text = f'{stat.require_channel}'
            self.label_remain_channel.text = f'{stat.remain_channel}'
            self.label_remain_electrode.text = f'{stat.remain_electrode}'


def electrode_efficient_npx(probe: NpxProbeDesp, chmap: ChannelMap, e: list[NpxElectrodeDesp]) -> ElectrodeEfficientStat:
    used_channel = len(chmap)
    used_channel_on_shanks = [
        len([it for it in chmap.electrodes if it.shank == s])
        for s in range(chmap.probe_type.n_shank)
    ]

    pp = len([it for it in e if it.policy == NpxProbeDesp.POLICY_SET])
    pf = len([it for it in e if it.policy == NpxProbeDesp.POLICY_D1])
    ph = len([it for it in e if it.policy == NpxProbeDesp.POLICY_D2])
    pq = len([it for it in e if it.policy == NpxProbeDesp.POLICY_D4])
    require_channels = pp + pf + ph / 2 + pq / 4

    pr = [it for it in e if it.policy in (NpxProbeDesp.POLICY_REMAINDER, NpxProbeDesp.POLICY_UNSET)]
    remain_channel = len([it for it in pr if it.state == NpxProbeDesp.STATE_USED])

    return ElectrodeEfficientStat(
        chmap.probe_type.n_channels,
        used_channel,
        used_channel_on_shanks,
        require_channels,
        remain_channel,
        len(pr)
    )
