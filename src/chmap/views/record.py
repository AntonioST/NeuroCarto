import time
from pathlib import Path

from .base import RecordStep, RecordView, R, ViewBase

__all__ = ['RecordManager', 'HistoryView']


class RecordManager:
    def __init__(self):
        self.views: list[RecordView] = []
        self.steps: list[RecordStep] = []
        self._is_replaying = False

    def register(self, view: RecordView):
        if view in self.views:
            return

        def add_record(record: R):
            if not self._is_replaying:
                self.add_record(view, record)

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

    def add_record(self, view: RecordView[R], record: R):
        self.steps.append(RecordStep(type(view).__name__, time.time(), record))

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

        self.steps = steps

    def save_steps(self, file: str | Path):
        import json

        data = [
            dict(source=it.source, time_stamp=it.time_stamp, record=it.record)
            for it in self.steps
        ]

        with Path(file).open('w') as f:
            json.dump(data, f, indent=2)


class HistoryView(ViewBase):

    @property
    def name(self) -> str:
        return 'History'
