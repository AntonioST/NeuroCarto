import abc
import functools
import inspect
from collections.abc import Callable
from typing import TypeVar

from bokeh.application.application import SessionContext
from bokeh.document import Document
from bokeh.model import Model
from bokeh.plotting import figure as Figure
from bokeh.server.server import Server

__all__ = [
    'BokehApplication',
    'run_later',
    'run_timeout',
    'run_periodic',
    'run_server',
    'Figure'
]

A = TypeVar('A')


class UIComponent(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def model(self) -> Model:
        pass


class BokehApplication(metaclass=abc.ABCMeta):
    document: Document

    @property
    def title(self) -> str:
        return type(self).__name__

    def setup(self, document: Document):
        self.document = document
        document.title = self.title

        document.add_root(self.index())
        document.on_session_destroyed(self.cleanup_session)
        document.add_next_tick_callback(self.update)

    @abc.abstractmethod
    def index(self) -> Model:
        pass

    def update(self):
        pass

    def cleanup_session(self, session_context: SessionContext):
        pass

    @classmethod
    def get_application(cls: type[A]) -> A:
        for frame in inspect.stack():
            if isinstance((app := frame.frame.f_locals.get('self', None)), BokehApplication):
                return app
        raise RuntimeError()

    def run_later(self, callback: Callable, *args, **kwargs):
        self.document.add_next_tick_callback(functools.partial(callback, *args, **kwargs))

    def run_timeout(self, delay: int, callback: Callable, *args, **kwargs):
        self.document.add_timeout_callback(functools.partial(callback, *args, **kwargs), delay)

    def run_periodic(self, cycle: int, callback: Callable, *args, **kwargs):
        self.document.add_periodic_callback(functools.partial(callback, *args, **kwargs), cycle)

    @classmethod
    def get_server(cls) -> Server:
        for frame in inspect.stack():
            if frame.function == 'run_server' and isinstance((server := frame.frame.f_locals.get('server', None)), Server):
                return server
        raise RuntimeError()


def run_later(callback: Callable, *args, **kwargs):
    # TODO bokeh.io.curdoc() ?
    document = BokehApplication.get_application().document
    document.add_next_tick_callback(functools.partial(callback, *args, **kwargs))


def run_timeout(delay: int, callback: Callable, *args, **kwargs):
    document = BokehApplication.get_application().document
    document.add_timeout_callback(functools.partial(callback, *args, **kwargs), delay)


def run_periodic(cycle: int, callback: Callable, *args, **kwargs):
    document = BokehApplication.get_application().document
    document.add_periodic_callback(functools.partial(callback, *args, **kwargs), cycle)


def run_server(handlers: BokehApplication | dict[str, BokehApplication], *,
               no_open_browser: bool = False):
    """
    :param handlers:
    :param no_open_browser:
    :return:
    """
    if isinstance(handlers, BokehApplication):
        handlers = {'/': handlers}

    if '/' not in handlers:
        raise RuntimeError("no '/' handler")

    server = Server({
        p: h.setup
        for p, h in handlers.items()
    }, num_procs=1)
    server.start()

    if not no_open_browser:
        server.io_loop.add_callback(server.show, "/")

    server.run_until_shutdown()
