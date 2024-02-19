import logging
import re
import time
from pathlib import Path

import numpy as np
from bokeh.models import DataTable, ColumnDataSource, TableColumn, TextInput, Div, CDSView, CustomJS, Toggle
from numpy.typing import NDArray

from chmap.config import ChannelMapEditorConfig, parse_cli
from chmap.util.bokeh_util import ButtonFactory, as_callback, new_help_button
from .base import RecordStep, RecordView, R, ViewBase, ControllerView

__all__ = ['RecordManager', 'HistoryView']


class RecordManager:
    def __init__(self):
        self.logger = logging.getLogger('chmap.history')

        self.views: list[RecordView] = []
        self.steps: list[RecordStep] = []
        self.disable = False
        self._is_replaying = False
        self._view: HistoryView | None = None

    def register(self, view: RecordView):
        if view in self.views:
            return

        self.logger.debug('register %s', type(view).__name__)

        def add_record(record: R, category: str, description: str):
            if not self._is_replaying and not self.disable:
                self._add_record(view, record, category, description)

        setattr(view, 'add_record', add_record)
        self.views.append(view)

    def unregister(self, view: RecordView):
        try:
            i = self.views.index(view)
        except ValueError:
            return
        else:
            self.logger.debug('unregister %s', type(view).__name__)
            del self.views[i]
            setattr(view, 'add_record', RecordView.add_record)

    def _add_record(self, view: RecordView[R], record: R, category: str, description: str):
        step = RecordStep(type(view).__name__, time.time(), category, description, record)
        self.logger.debug('get %s', step)
        self.steps.append(step)

        if (history := self._view) is not None:
            history.on_add_record(step)

    def replay(self, *, reset=False, index: list[int] = None):
        """

        :param reset: Is it a reset replay?
        :param index: only replay steps on given index.
        :return:
        """
        if self._is_replaying:
            raise RuntimeError()

        self._is_replaying = True
        try:
            if index is None:
                self.logger.debug('replay all steps')
                steps = self.steps
            else:
                self.logger.debug('replay steps on %s', index)
                steps = [self.steps[it] for it in index]

            for view in self.views:
                view.replay_records(steps, reset=reset)
        finally:
            self._is_replaying = False

    def load_steps(self, file: str | Path, *, blacklist: list[str] = tuple()):
        self.logger.debug('load history %s', file)
        self.steps = self._load_steps(file, blacklist=blacklist)

    def _load_steps(self, file: str | Path, *, blacklist: list[str] = tuple()) -> list[RecordStep]:
        import json
        with Path(file).open() as f:
            data = json.load(f)

        steps = []
        for item in data:
            if item['source'] not in blacklist:
                steps.append(RecordStep.from_dict(item))

        return steps

    def save_steps(self, file: str | Path, append=False):
        self.logger.debug('save history %s', file)

        import json

        data = []
        if append and file.exists():
            data.extend(self._load_steps(file))

        data.extend([it.as_dict() for it in self.steps])

        with Path(file).open('w') as f:
            json.dump(data, f, indent=2)


class HistoryView(ViewBase, ControllerView):
    history_step_data: ColumnDataSource
    history_step_view: CDSView

    def __init__(self, config: ChannelMapEditorConfig):
        super().__init__(config, logger='chmap.view.history')
        self.manager: RecordManager | None = None
        self.history_step_data = ColumnDataSource(data=dict(source=[], category=[], action=[]))
        self.history_step_view = CDSView()

    @property
    def name(self) -> str:
        return 'History'

    def get_history_file(self) -> Path:  # TODO make it selectable
        return self.get_app().cache_file('history.json')

    def load_history(self):
        if (manager := self.manager) is not None and (history_file := self.get_history_file()).exists():
            manager.load_steps(history_file)
            self.update_history_table()

    def save_history(self):
        if (manager := self.manager) is not None:
            history_file = self.get_history_file()
            history_file.parent.mkdir(parents=True, exist_ok=True)
            manager.save_steps(history_file, append=False)
            self.logger.debug(f'save history : %s', history_file)

    def on_add_record(self, record: RecordStep):
        self.history_step_data.stream(dict(
            source=[record.source],
            category=[record.category],
            action=[record.description]
        ))

        self.set_status(f'history : {len(self.manager.steps)}', decay=10)

    def update_history_table(self):
        if (manager := self.manager) is None:
            self.history_step_data.data = dict(source=[], category=[], action=[])
            return

        source = []
        category = []
        action = []
        for i, record in enumerate(manager.steps):
            source.append(record.source)
            category.append(record.category)
            action.append(record.description)

        self.history_step_data.data = dict(source=source, category=category, action=action)

    # ============= #
    # UI components #
    # ============= #

    history_step_table: DataTable
    source_filter: TextInput
    category_filter: TextInput
    disable_toggle: Toggle
    description_filter: TextInput

    def _setup_content(self, **kwargs):
        new_btn = ButtonFactory(min_width=100, width_policy='min')

        self.history_step_table = DataTable(
            source=self.history_step_data,
            view=self.history_step_view,
            columns=[
                TableColumn(field='source', title='Source', width=100),
                TableColumn(field='category', title='Category', width=100),
                TableColumn(field='action', title='Action'),
            ],
            width=500, height=300, reorderable=False, sortable=False,
        )

        self.source_filter = TextInput(width=100)
        self.source_filter.on_change('value', as_callback(self._on_filter_update))
        self.category_filter = TextInput(width=100)
        self.category_filter.on_change('value', as_callback(self._on_filter_update))
        self.description_filter = TextInput(width=200)
        self.description_filter.on_change('value', as_callback(self._on_filter_update))

        replay_callback = TextInput(visible=False)
        replay_callback.on_change('value', as_callback(self._on_replay))

        # https://discourse.bokeh.org/t/get-the-number-of-elements-returned-by-a-cdsview-in-bokeh/11206/4
        replay = new_btn('Replay', None)
        replay.js_on_click(handler=CustomJS(args=dict(view=self.history_step_view, callback=replay_callback), code="""
        callback.value = view._indices.join(',');
        """))

        self.disable_toggle = Toggle(label='Disable', min_width=100, width_policy='min')
        self.disable_toggle.on_change('active', as_callback(self._on_disable))

        from bokeh.layouts import row, column
        return [
            row(
                self.history_step_table,
                column(
                    replay,
                    new_btn('Save', self.save_history),
                    new_btn('Load', self.load_history),
                    new_btn('Delete', self.on_delete),
                    new_btn('Clear', self.on_clear),
                    self.disable_toggle,
                    replay_callback
                )
            ),
            row(Div(text='<b>Filter</b>'),
                self.source_filter, self.category_filter,
                new_help_button("Source and Category filter. Match exactly. Use ',' to select multiple", position='top'),
                self.description_filter,
                new_help_button("Action filter. Match Any. Use '~', '&' or '|' to do selection inverse, intersect or union, respectively.", position='top')),
        ]

    def _on_filter_update(self):
        from bokeh.models import filters

        value: str
        selectors = []
        if len(value := self.source_filter.value.strip()) > 0:
            selectors.append(self._on_filter_category('source', value))

        if len(value := self.category_filter.value.strip()) > 0:
            selectors.append(self._on_filter_category('category', value))

        if len(value := self.description_filter.value.strip()) > 0:
            description = list(self.history_step_data.data['action'])
            booleans = self._on_filter_description(description, value)
            selectors.append(filters.BooleanFilter(booleans=list(booleans)))

        if len(selectors) == 0:
            self.history_step_view.filter = filters.AllIndices()
        elif len(selectors) == 1:
            self.history_step_view.filter = selectors[0]
        else:
            self.history_step_view.filter = filters.IntersectionFilter(operands=selectors)

    def _on_filter_category(self, column: str, value: str):
        from bokeh.models import filters

        if ',' in value:
            return filters.UnionFilter(operands=[
                filters.GroupFilter(column_name=column, group=it.strip())
                for it in value.split(',')
            ])
        else:
            return filters.GroupFilter(column_name=column, group=value)

    def _on_filter_description(self, description: list[str], expr: str) -> NDArray[np.bool_]:
        old_expr = expr
        expr = re.sub(r'([^ \t~|&^()]+)', r'_a("\1")', expr)
        self.logger.debug('filter description "%s" -> "%s"', old_expr, expr)

        def _a(word: str) -> NDArray[np.bool_]:
            word = word.lower()
            return np.array([word in it.lower() for it in description])

        return eval(expr, {}, dict(_a=_a))

    def _on_replay(self, value: str):
        if (manager := self.manager) is None:
            return

        index = list(map(int, value.split(',')))

        manager.replay(reset=True, index=index)
        self.get_app().on_probe_update()

    def on_replay(self):
        if (manager := self.manager) is None:
            return

        manager.replay(reset=True)
        self.get_app().on_probe_update()

    def on_delete(self):
        if (manager := self.manager) is None:
            return

        selected = set(self.history_step_data.selected.indices)
        manager.steps = [step for i, step in enumerate(manager.steps) if i not in selected]
        self.update_history_table()

    def on_clear(self):
        if (manager := self.manager) is None:
            return

        manager.steps = []
        self.update_history_table()

    def _on_disable(self, disable: bool):
        if (manager := self.manager) is not None:
            manager.disable = disable

    # ============== #
    # update methods #
    # ============== #

    def start(self):
        self.manager = self.get_app().record_manager
        if self.manager is None:
            self.log_message('app history feature is disabled')
        else:
            self.manager._view = self
            self.update_history_table()


if __name__ == '__main__':
    import sys

    from chmap.main_bokeh import main

    main(parse_cli([
        *sys.argv[1:],
        '-C', 'res',
        '--debug',
        '--view=-',
        '--view=chmap.views.record:HistoryView',
    ]))
