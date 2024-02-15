import time
from pathlib import Path

from chmap.config import ChannelMapEditorConfig
from .base import RecordStep, RecordView, R, ViewBase, ControllerView, GlobalStateView

__all__ = ['RecordManager', 'HistoryView']


class RecordManager:
    def __init__(self):
        self.views: list[RecordView] = []
        self.steps: list[RecordStep] = []
        self._is_replaying = False
        self._view: HistoryView | None = None

    def register(self, view: RecordView):
        if view in self.views:
            return

        def add_record(record: R):
            if not self._is_replaying:
                self._add_record(view, record)

        setattr(view, 'add_record', add_record)
        self.views.append(view)

    def unregister(self, view: RecordView):
        try:
            i = self.views.index(view)
        except ValueError:
            return
        else:
            del self.views[i]
            setattr(view, 'add_record', RecordView.add_record)

    def _add_record(self, view: RecordView[R], record: R):
        step = RecordStep(type(view).__name__, time.time(), record)
        self.steps.append(step)

        if (history := self._view) is not None:
            history.on_add_record(step)

    def replay(self, reset=False):
        if self._is_replaying:
            raise RuntimeError()

        self._is_replaying = True
        try:
            for view in self.views:
                view.replay_records(self.steps, reset=reset)
        finally:
            self._is_replaying = False

    def load_steps(self, file: str | Path, *, blacklist: list[str] = tuple()):
        self.steps = self._load_steps(file, blacklist=blacklist)

    def _load_steps(self, file: str | Path, *, blacklist: list[str] = tuple()) -> list[RecordStep]:
        import json
        with Path(file).open() as f:
            data = json.load(f)

        steps = []
        for item in data:
            source = item['source']
            if source not in blacklist:
                steps.append(
                    RecordStep(
                        source,
                        item['time_stamp'],
                        item['record'],
                    )
                )

        return steps

    def save_steps(self, file: str | Path, append=False):
        import json

        data = []
        if append:
            data.extend(self._load_steps(file))

        data.extend([
            dict(source=it.source, time_stamp=it.time_stamp, record=it.record)
            for it in self.steps
        ])

        with Path(file).open('w') as f:
            json.dump(data, f, indent=2)


class HistoryView(ViewBase, ControllerView, GlobalStateView):

    def __init__(self, config: ChannelMapEditorConfig):
        super().__init__(config, logger='chmap.view.history')
        self.manager: RecordManager | None = None

    @property
    def name(self) -> str:
        return 'History'

    def get_history_file(self) -> Path:
        return self.get_app().cache_file('history.json')

    def load_history(self):
        if (manager := self.manager) is not None and (history_file := self.get_history_file()).exists():
            manager.load_steps(history_file)

    def save_history(self):
        if (manager := self.manager) is not None:
            history_file = self.get_history_file()
            history_file.parent.mkdir(parents=True, exist_ok=True)
            manager.save_steps(history_file, append=True)
            self.logger.debug(f'save history : %s', history_file)

    # ============= #
    # UI components #
    # ============= #

    def _setup_content(self, **kwargs):
        pass

    def on_clear(self):
        pass

    # ========= #
    # load/save #
    # ========= #

    def save_state(self, local=True):
        if not local:
            self.save_history()

    def restore_state(self, state):
        # do nothing
        pass

    # ============== #
    # update methods #
    # ============== #

    def start(self):
        self.manager = self.get_app().record_manager
        if self.manager is None:
            self.log_message('app history feature is disabled')
        else:
            self.manager._view = self

    # ============== #
    # notify methods #
    # ============== #

    def on_add_record(self, step: RecordStep):
        self.logger.debug('add record from %s', step.source)
        self.set_status(f'history : {len(self.manager.steps)}', decay=10)
