import time
from collections.abc import Iterable

from bokeh.models import ColumnDataSource, GlyphRenderer
from bokeh.plotting import figure as Figure

from chmap.probe import ProbeDesp, E, M

__all__ = ['ProbeView']


class ProbeView:
    data_electrodes: dict[int, ColumnDataSource]
    render_electrodes: dict[int, GlyphRenderer]

    data_highlight: ColumnDataSource
    render_highlight: GlyphRenderer

    def __init__(self, desp: ProbeDesp[M, E]):
        self.probe: ProbeDesp[M, E] = desp
        self.channelmap: M | None = None
        self.electrodes: list[E] | None = None
        self._e2i: dict[E, int] = {}

        self.data_electrodes = {}
        for state in self.probe.possible_states.values():
            self.data_electrodes[state] = ColumnDataSource(data=dict(x=[], y=[], e=[]))
            self.data_electrodes[state].selected.on_change('indices', self.on_select(state))
        if (state := ProbeDesp.STATE_FORBIDDEN) not in self.data_electrodes:
            self.data_electrodes[state] = ColumnDataSource(data=dict(x=[], y=[], e=[]))
            self.data_electrodes[state].selected.on_change('indices', self.on_select(state))
        self.data_highlight = ColumnDataSource(data=dict(x=[], y=[], e=[]))

        self.style_electrodes = {  # TODO config somewhere
            ProbeDesp.STATE_USED: dict(color='green'),
            ProbeDesp.STATE_UNUSED: dict(color='black'),
            ProbeDesp.STATE_FORBIDDEN: dict(color='red', size=2, alpha=0.2),
            'highlight': dict(color='yellow', size=6, alpha=0.5)
        }

    def plot(self, f: Figure):
        self.render_electrodes = {}
        self.render_highlight = f.scatter(
            x='x', y='y', source=self.data_highlight, **self.style_electrodes.get('highlight', {})
        )

        for state, data in self.data_electrodes.items():
            self.render_electrodes[state] = f.scatter(
                x='x', y='y', source=data, **self.style_electrodes.get(state, {})
            )

    def channelmap_desp(self) -> str:
        return self.probe.channelmap_desp(self.channelmap)

    def reset(self, *args, **kwargs):
        if len(args) == 0 and len(kwargs) == 0 and self.channelmap is not None:
            channelmap = self.probe.new_channelmap(self.channelmap)
        else:
            channelmap = self.probe.new_channelmap(*args, **kwargs)

        self.channelmap = channelmap
        self.electrodes = self.probe.all_electrodes(channelmap)
        self._reset_electrode_state()

        self._e2i = {}
        for i, e in enumerate(self.electrodes):  # type: int, E
            self._e2i[e] = i

    def _reset_electrode_state(self):
        for e in self.electrodes:
            e.state = ProbeDesp.STATE_UNUSED

        c = self.probe.all_channels(self.channelmap, self.electrodes)
        for e in self.probe.invalid_electrodes(self.channelmap, c, self.electrodes):
            e.state = ProbeDesp.STATE_FORBIDDEN
        for e in c:
            e.state = ProbeDesp.STATE_USED

    def update_electrode(self):
        for state, data in self.data_electrodes.items():
            self.update_electrode_position(data, self.get_electrodes(None, state=state))
        self.update_electrode_position(self.data_highlight, [])

    def refresh_selection(self):
        self.channelmap = self.probe.select_electrodes(self.channelmap, self.electrodes)
        self._reset_electrode_state()

    def get_electrodes(self, s: None | int | list[int] | ColumnDataSource, *, state: int = None) -> list[E]:
        ret: list[E]

        match s:
            case None:
                ret = list(self.electrodes)
            case int():
                ret = [self.electrodes[s]]
            case list():
                ret = [self.electrodes[it] for it in s]
            case _ if isinstance(s, ColumnDataSource):
                ret = [self.electrodes[it] for it in s.data['e']]
            case _:
                raise TypeError()

        if state is not None:
            ret = [it for it in ret if it.state == state]

        return ret

    def get_selected(self, d: ColumnDataSource = None, *, reset=False) -> set[E]:
        if d is None:
            ret = set()
            for state, data in self.data_electrodes.items():
                ret.update(self.get_selected(data, reset=reset))
            return ret
        else:
            selected_index = d.selected.indices
            if reset:
                d.selected.indices = []

            e = d.data['e']
            return set(self.get_electrodes([e[it] for it in selected_index]))

    def on_select(self, state: int):
        time_stamp = 0

        # noinspection PyUnusedLocal
        def on_select_callback(prop: str, old: list[int], selected: list[int]):
            nonlocal time_stamp
            now = time.time()
            self.set_highlight(self.get_selected(self.data_electrodes[state]), append=now - time_stamp < 1)
            time_stamp = now

        return on_select_callback

    def update_electrode_position(self, d: ColumnDataSource, e: Iterable[E], *, append=False):
        x = [it.x for it in e]
        y = [it.y for it in e]
        i = [self._e2i[it] for it in e]

        if append:
            data = d.data
            x.extend(data['x'])
            y.extend(data['y'])
            i.extend(data['e'])

        d.data = dict(x=x, y=y, e=i)

    def set_highlight(self, s: Iterable[E], *, invalid=True, append=False):
        h = list(s)

        if invalid:
            h = self.probe.invalid_electrodes(self.channelmap, h, self.electrodes)

        self.update_electrode_position(self.data_highlight, h, append=append)

    def set_state_for_selected(self, state: int):
        if state == ProbeDesp.STATE_USED:
            for e in self.get_selected(self.data_electrodes[ProbeDesp.STATE_UNUSED], reset=True):
                self.probe.add_electrode(self.channelmap, e)
            for e in self.get_selected(self.data_electrodes[ProbeDesp.STATE_FORBIDDEN], reset=True):
                self.probe.add_electrode(self.channelmap, e, overwrite=True)

        elif state == ProbeDesp.STATE_UNUSED:
            for e in self.get_selected(self.data_electrodes[ProbeDesp.STATE_UNUSED], reset=True):
                self.probe.del_electrode(self.channelmap, e)

        self._reset_electrode_state()

    def set_policy_for_selected(self, policy: int):
        for e in self.get_selected(reset=True):
            e.policy = policy
