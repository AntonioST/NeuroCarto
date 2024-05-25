from __future__ import annotations

from collections.abc import Iterable
from typing import Any, TypedDict, Generic, TYPE_CHECKING

from bokeh.models import ColumnDataSource, GlyphRenderer, tools, UIElement, Div
from neurocarto.config import CartoConfig
from neurocarto.probe import ProbeDesp, M, E
from neurocarto.util.bokeh_app import run_timeout
from neurocarto.util.bokeh_util import as_callback
from neurocarto.util.debug import TimeMarker
from neurocarto.util.utils import doc_link
from neurocarto.views.base import Figure, ViewBase, RecordView, RecordStep
from typing_extensions import Required

if TYPE_CHECKING:
    from bokeh.server.callbacks import TimeoutCallback

__all__ = ['ProbeView']


class ProbeViewAction(TypedDict, total=False):
    action: Required[str]  # reset, set_state, set_category, channelmap, blueprint

    # action=reset
    code: int

    # action=set_state, set_category: captured electrode id
    # action=channelmap: selected electrode id
    # action=blueprint: all electrode's category values
    electrodes: list[int]

    # action=set_state
    state: int

    # action=set_category
    category: int


class ProbeView(Generic[M, E], ViewBase, RecordView[ProbeViewAction]):
    """
    Probe view.
    """

    STYLES: dict[int | str, dict[str, Any]] = {
        ProbeDesp.STATE_USED: dict(color='green'),
        ProbeDesp.STATE_UNUSED: dict(color='black'),
        ProbeDesp.STATE_DISABLED: dict(color='red', size=2, alpha=0.2),
        'highlight': dict(color='yellow', size=6, alpha=0.5)
    }

    data_electrodes: dict[int, ColumnDataSource]
    render_electrodes: dict[int, GlyphRenderer]

    data_highlight: ColumnDataSource
    render_highlight: GlyphRenderer

    def __init__(self, config: CartoConfig, desp: ProbeDesp[M, E]):
        super().__init__(config, logger='neurocarto.view.probe')

        self.logger.debug('init(%s)', type(desp).__name__)
        self.probe: ProbeDesp[M, E] = desp
        self.channelmap: M | None = None
        self.electrodes: list[E] | None = None
        self._e2i: dict[E, int] = {}  # {E: electrode_index}

        self.data_electrodes = {}  # {state : ColumnDataSource}
        """dict {state: ColumnDataSource}"""

        for state in (ProbeDesp.STATE_UNUSED, ProbeDesp.STATE_USED, ProbeDesp.STATE_DISABLED):
            self.data_electrodes[state] = ColumnDataSource(data=dict(x=[], y=[], e=[], c=[]))
            self.data_electrodes[state].selected.on_change('indices', as_callback(self._on_capture, state=state))

        self.data_highlight = ColumnDataSource(data=dict(x=[], y=[], e=[]))

        self.logger.debug('use selector(%s)', config.selector)
        self.selecting_parameters = {
            'selector': config.selector
        }

    @property
    def name(self) -> str:
        return self.probe.channelmap_desp(self.channelmap)

    # ============= #
    # UI components #
    # ============= #

    def setup(self, f: Figure, **kwargs) -> list[UIElement]:
        self._setup_render(f, **kwargs)

        from bokeh.layouts import row
        return [row(self._setup_title(**kwargs))]

    def _setup_render(self, f: Figure, **kwargs):
        self.logger.debug('setup(figure)')
        self.render_highlight = f.scatter(
            x='x', y='y', source=self.data_highlight, **self.STYLES.get('highlight', {})
        )

        self.render_electrodes = {
            state: f.scatter(
                x='x', y='y', source=data, **self.STYLES.get(state, {})
            )
            for state, data in self.data_electrodes.items()
        }

        # toolbar
        f.tools.insert(2, tools.BoxSelectTool(
            description='select electrode',
            renderers=list(self.render_electrodes.values())
        ))

        f.tools.append(tools.HoverTool(
            description="electrode information",
            renderers=[
                self.render_electrodes[ProbeDesp.STATE_USED],
                self.render_electrodes[ProbeDesp.STATE_UNUSED],
                self.render_electrodes[ProbeDesp.STATE_DISABLED],
            ],
            tooltips=[  # hover
                ('Channel', "@c"),
                ("(x,y)", "($x, $y)"),
            ]
        ))

    def _setup_title(self, **kwargs) -> list[UIElement]:
        self.view_title = Div(text=f'<b>{self.name}</b>')
        self.status_div = Div(text='')

        return [
            self.view_title, self.status_div
        ]

    def _setup_content(self, **kwargs):
        raise RuntimeError()

    def update_probe_desp(self):
        self.view_title.text = f'<b>{self.name}<b/>'

    # ======= #
    # actions #
    # ======= #

    def reset(self, chmap: int | M | None = None):
        """
        Reset channelmap.

        :param chmap: channelmap code
        """
        if chmap is None:
            if (current := self.channelmap) is not None:
                self.logger.debug('reset()')
                channelmap = self.probe.new_channelmap(current)
            else:
                return

        elif isinstance(chmap, int):
            self.logger.debug('reset(%d)', chmap)
            channelmap = self.probe.new_channelmap(chmap)
        else:
            self.logger.debug('reset(%s)', type(chmap).__name__)
            channelmap = self.probe.copy_channelmap(chmap)

        self.channelmap = channelmap
        self.electrodes = self.probe.all_electrodes(channelmap)
        self._reset_electrode_state()

        self._e2i = {}
        for i, e in enumerate(self.electrodes):  # type: int, E
            self._e2i[e] = i

        if (code := self.probe.channelmap_code(self.channelmap)) is None:
            raise RuntimeError('un-reachable')

        desp = f'create new {self.probe.type_description(code)}'
        self.add_record(ProbeViewAction(action='reset', code=code), 'reset', desp)

    def _reset_electrode_state(self):
        for e in self.electrodes:
            e.state = ProbeDesp.STATE_UNUSED

        c = self.probe.all_channels(self.channelmap, self.electrodes)
        for e in self.probe.invalid_electrodes(self.channelmap, c, self.electrodes):
            e.state = ProbeDesp.STATE_DISABLED
        for e in c:
            e.state = ProbeDesp.STATE_USED

        self.update_probe_desp()

    def update_electrode(self):
        """Refresh channelmap"""
        for state, data in self.data_electrodes.items():
            self.update_electrode_position(data, self.get_electrodes(None, state=state))
        self.update_electrode_position(self.data_highlight, [])
        self.update_probe_desp()

    def refresh_selection(self):
        """Rerun electrode selection and refresh channelmap"""
        if self.channelmap is None:
            return

        electrodes = [
            e.category
            for e in self.electrodes
        ]
        self.add_record(ProbeViewAction(action='blueprint', electrodes=electrodes),
                        'selection', 'electrode selection blueprint')

        self.logger.debug('refresh_selection()')
        try:
            mark = TimeMarker()
            self.channelmap = self.probe.select_electrodes(self.channelmap, self.electrodes, **self.selecting_parameters)
            t = mark()

            self.logger.debug('refresh_selection() used %.2f sec', t)
        except BaseException as e:
            self.logger.warning('refresh_selection() fail', exc_info=e)
            self.log_message('refresh fail')
        else:
            self._reset_electrode_state()

            electrodes = [
                i
                for i, e in enumerate(self.electrodes)
                if e.state == ProbeDesp.STATE_USED
            ]
            self.add_record(ProbeViewAction(action='channelmap', electrodes=electrodes),
                            'selection', 'electrode selection result')

    def get_electrodes(self, s: None | int | list[int] | ColumnDataSource, *, state: int = None) -> list[E]:
        """
        Get electrodes in source *s*.

        :param s: electrode source.
        :param state: filter electrodes with its state in returns
        :return: electrodes
        """
        if (electrodes := self.electrodes) is None:
            return []

        ret: list[E]

        match s:
            case None:
                ret = list(electrodes)
            case int():
                ret = [electrodes[s]]
            case list():
                ret = [electrodes[it] for it in s]
            case _ if isinstance(s, ColumnDataSource):
                ret = [electrodes[it] for it in s.data['e']]  # type: ignore
            case _:
                raise TypeError()

        if state is not None:
            ret = [it for it in ret if it.state == state]

        return ret

    @doc_link()
    def get_captured_electrodes(self, d: ColumnDataSource = None, *, reset=False) -> set[E]:
        """
        Get captured electrodes.

        :param d: one of {#data_electrodes}
        :param reset: reset the selecting state.
        :return: captured electrodes.
        """
        return set(self.get_electrodes(self.get_captured_electrodes_index(d, reset=reset)))

    def get_captured_electrodes_index(self, d: ColumnDataSource = None, *, reset=False) -> list[int]:
        """
        Get captured electrodes.

        :param d: one of {#data_electrodes}
        :param reset: reset the selecting state.
        :return: index-list of captured electrodes.
        """
        if d is None:
            ret = list[int]()
            for state, data in self.data_electrodes.items():
                ret.extend(self.get_captured_electrodes_index(data, reset=reset))
            return ret
        else:
            selected_index = d.selected.indices
            if reset:
                d.selected.indices = []

            e = d.data['e']
            return [e[it] for it in selected_index]

    def set_capture_electrodes(self, electrodes: list[int] | list[E], d: ColumnDataSource = None):
        if d is None:
            for data in self.data_electrodes.values():
                self.set_capture_electrodes(electrodes, data)
        else:
            i = set([
                it if isinstance(it, int) else self._e2i[it]
                for it in electrodes
            ])

            e = d.data['e']
            s = [ii for ii, ie in enumerate(e) if ie in i]
            d.selected.indices = s

    def clear_capture_electrode(self, d: ColumnDataSource = None):
        if d is None:
            for data in self.data_electrodes.values():
                self.clear_capture_electrode(data)
        else:
            d.selected.indices = []

    _captured_electrodes: list[E] = []
    _captured_callback: TimeoutCallback | None = None

    def _on_capture(self, state: int):
        selected = self.get_captured_electrodes(self.data_electrodes[state])
        self._captured_electrodes.extend(selected)
        if self._captured_callback is None:
            self._captured_callback = run_timeout(100, self._on_capture_callback)

    def _on_capture_callback(self):
        selected = self._captured_electrodes
        self._captured_electrodes = []
        self._captured_callback = None

        if len(selected):
            self.logger.debug('select %d electrodes', len(selected))
        self.set_highlight(selected)

    @doc_link()
    def update_electrode_position(self, d: ColumnDataSource, e: Iterable[E]):
        """
        Show electrodes.

        :param d: one of {#data_electrodes}
        :param e: new electrodes
        """
        x = []
        y = []
        i = []
        c = []

        for it in e:
            x.append(it.x)
            y.append(it.y)
            i.append(self._e2i[it])
            c.append(str(it.channel))

        d.data = dict(x=x, y=y, e=i, c=c)

    def set_highlight(self, s: Iterable[E], *, invalid=True):
        """
        Highlight electrodes.

        :param s: selected electrode set
        :param invalid: extend the highlight set to include co-electrodes
        """
        if invalid and (current := self.channelmap) is not None:
            assert self.electrodes is not None
            s = self.probe.invalid_electrodes(current, s, self.electrodes)

        self.update_electrode_position(self.data_highlight, s)

    @doc_link()
    def set_state_for_captured(self, state: int, electrodes: list[int | E] = None):
        """
        Set electrode state for selected electrodes.

        :param state: new state. value in {ProbeDesp#STATE_USED}, {ProbeDesp#STATE_UNUSED}
        :param electrodes: captured electrodes.
        """
        if (channelmap := self.channelmap) is None:
            return

        if state not in (ProbeDesp.STATE_USED, ProbeDesp.STATE_UNUSED):
            return

        captured = []
        if electrodes is not None:
            for e in electrodes:
                if isinstance(e, int):
                    if (_electrodes := self.electrodes) is None:
                        return
                    i, e = e, _electrodes[e]
                else:
                    i, e = self._e2i[e], e

                if state == ProbeDesp.STATE_USED:
                    self.probe.add_electrode(channelmap, e, overwrite=True)
                elif state == ProbeDesp.STATE_UNUSED:
                    self.probe.del_electrode(channelmap, e)

                captured.append(i)

        elif state == ProbeDesp.STATE_USED:
            for e in self.get_captured_electrodes(self.data_electrodes[ProbeDesp.STATE_UNUSED], reset=True):
                captured.append(self._e2i[e])
                self.probe.add_electrode(channelmap, e)
            for e in self.get_captured_electrodes(self.data_electrodes[ProbeDesp.STATE_DISABLED], reset=True):
                captured.append(self._e2i[e])
                self.probe.add_electrode(channelmap, e, overwrite=True)
            for e in self.get_captured_electrodes(self.data_electrodes[ProbeDesp.STATE_USED], reset=True):
                captured.append(self._e2i[e])

        elif state == ProbeDesp.STATE_UNUSED:
            for e in self.get_captured_electrodes(self.data_electrodes[ProbeDesp.STATE_USED], reset=True):
                captured.append(self._e2i[e])
                self.probe.del_electrode(channelmap, e)
            for e in self.get_captured_electrodes(self.data_electrodes[ProbeDesp.STATE_UNUSED], reset=True):
                captured.append(self._e2i[e])
            for e in self.get_captured_electrodes(self.data_electrodes[ProbeDesp.STATE_DISABLED], reset=True):
                captured.append(self._e2i[e])

        self._reset_electrode_state()
        if len(captured) > 0:
            if (state_name := self.probe.state_description(state)) is None:
                state_name = f'State[{state}]'

            self.add_record(ProbeViewAction(action='set_state', electrodes=captured, state=state),
                            'state', f'set {len(captured)} electrodes to {state_name}')

    @doc_link()
    def set_category_for_captured(self, category: int, electrodes: list[int | E] = None):
        """
        Set electrode category value for selected electrodes.

        :param category: category value from {ProbeDesp}.CATE_*
        :param electrodes: captured electrodes.
        """
        if (current := self.electrodes) is None:
            return []

        captured = []
        if electrodes is not None:
            for e in electrodes:
                if isinstance(e, int):
                    i, e = e, current[e]
                else:
                    i, e = self._e2i[e], e

                e.category = category
                captured.append(i)
        else:
            for e in self.get_captured_electrodes(reset=True):
                e.category = category
                captured.append(self._e2i[e])

        if len(captured) > 0:
            if (category_name := self.probe.category_description(category)) is None:
                category_name = f'Category[{category}]'

            self.add_record(ProbeViewAction(action='set_category', electrodes=captured, category=category),
                            'category', f'set {len(captured)} electrodes to {category_name}')

    # ============ #
    # replay steps #
    # ============ #

    def filter_records(self, records: list[RecordStep], *, reset=False) -> list[RecordStep]:
        source = type(self).__name__

        ret = []
        prev_blueprint = None
        prev_channelmap = None

        for record in records:
            if record.source == source:
                match record.record:
                    case {'action': 'reset'}:
                        ret = [record]
                        prev_blueprint = None
                        prev_channelmap = None

                    case {'action': 'blueprint'}:
                        # only keep last one
                        if prev_blueprint is not None:
                            ret.remove(prev_blueprint)
                        ret.append(record)
                        prev_blueprint = record

                        # remove all blueprint setting
                        for i in reversed(range(len(ret))):
                            match ret[i]:
                                case {'action': 'set_category'}:
                                    del ret[i]

                    case {'action': 'channelmap'}:
                        # only keep last one
                        if prev_channelmap is not None:
                            ret.remove(prev_channelmap)

                        ret.append(record)
                        prev_channelmap = record

                        # remove all state setting
                        for i in reversed(range(len(ret))):
                            match ret[i]:
                                case {'action': 'set_state'}:
                                    del ret[i]

                    case _:
                        ret.append(record)

        return ret

    def replay_record(self, record: RecordStep):
        self.logger.debug('replay %s', record.description)

        match record.record:
            case {'action': 'reset', 'code': code}:
                self.reset(code)
            case {'action': 'set_state', 'electrodes': electrodes, 'state': state}:
                self.set_state_for_captured(state, electrodes)
            case {'action': 'set_category', 'electrodes': electrodes, 'category': category}:
                self.set_category_for_captured(category, electrodes)
            case {'action': 'channelmap', 'electrodes': electrodes}:
                if (channelmap := self.channelmap) is not None:
                    assert self.electrodes is not None

                    self.reset(channelmap)
                    for i in electrodes:  # type: int
                        self.probe.add_electrode(channelmap, self.electrodes[i], overwrite=True)
                    self._reset_electrode_state()

            case {'action': 'blueprint', 'electrodes': categories}:
                if (electrodes := self.electrodes) is not None:
                    for e, c in zip(electrodes, categories):  # type: E, int
                        e.category = c
