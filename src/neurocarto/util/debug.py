import inspect
import time
from pathlib import Path

__all__ = ['print_save', 'line_mark', 'TimeMarker']


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
