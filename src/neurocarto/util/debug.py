import inspect
import os
import time
from pathlib import Path

__all__ = ['print_save', 'line_mark', 'TimeMarker', 'Profiler']


def print_save(file: str) -> Path:
    """
    print the saving message for the *file*, which is used in a saving function call.

    It is a debugging use function.

    :param file:
    :return: Path of *file*
    """
    print('SAVE', file)
    return Path(file)


def line_mark(message: str):
    """
    print the current line number.

    It is a debugging use function.

    :param message:
    """
    frame = inspect.stack()[1]
    filename = frame.filename
    try:
        filename = filename[filename.index('neurocarto/'):]
    except ValueError:
        pass

    filename = filename.replace('.py', '').replace('/', '.')
    print(filename, f'@{frame.lineno}', f'{frame.function}()', '::', message)


class TimeMarker:
    """
    print the time interval between calls.

    It is a debugging use function.
    """

    def __init__(self, disable=False):
        self.t = time.time()
        self.disable = disable

    def reset(self):
        self.t = time.time()

    def __call__(self, message: str = None) -> float:
        t = time.time()
        d = t - self.t
        self.t = t

        if message is not None and not self.disable:
            print(message, f'use {d:.2f}')

        return d


class Profiler:
    def __init__(self, file: str | Path, *,
                 enable: bool | str = True,
                 capture_exception=False,
                 dump_on_exit=True):
        """

        :param file: stat dump file
        :param enable: enable profile. Use string, then this flag is controlled by the environment variable.
        :param capture_exception: capture exception when ``__exit__``.
        :param dump_on_exit: dump file when ``__exit__`` if enabled.
        """
        import cProfile

        if isinstance(enable, str):
            enable = len(os.getenv(enable, '')) > 0

        self.file = file
        self.enable = enable
        self.capture_exception = capture_exception
        self.dump_on_exit = dump_on_exit

        self.profile = cProfile.Profile()
        self.start_time: float | None = None
        self.duration: float = 0
        self.repeat: int = 0
        self.exception: BaseException | None = None

    def __enter__(self):
        if self.enable:
            self.start_time = time.time()
            self.profile.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            self.profile.disable()
            t = time.time() - self.start_time
            self.duration += t
            self.repeat += 1

            if self.dump_on_exit:
                self.profile.dump_stats(self.file)

        self.exception = exc_val
        if self.capture_exception:
            return True

    def dump_file(self) -> Path | None:
        self.profile.dump_stats(self.file)
        return self.file
