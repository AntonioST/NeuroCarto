import abc
import functools
import inspect
import sys
from collections.abc import Callable

from bokeh.application.application import SessionContext
from bokeh.document import Document
from bokeh.model import Model
from bokeh.plotting import figure as Figure
from bokeh.server.callbacks import SessionCallback
from bokeh.server.server import Server

if sys.version_info >= (3, 11):
    from typing import Self
else:
    from typing_extensions import Self

__all__ = [
    'BokehApplication',
    'run_later',
    'run_timeout',
    'run_periodic',
    'run_server',
    'Figure'
]


class BokehApplication(metaclass=abc.ABCMeta):
    """Bokeh Application of a single page"""

    document: Document

    @property
    def title(self) -> str:
        """Application title (shown in web page title)"""
        return type(self).__name__

    @abc.abstractmethod
    def index(self) -> Model:
        """Web-page document content"""
        pass

    def start(self):
        """Invoked when session set"""
        pass

    def cleanup(self, context: SessionContext):
        """Invoked when session destroyed"""
        pass

    @classmethod
    def get_application(cls) -> Self:
        for frame in inspect.stack():
            if isinstance((app := frame.frame.f_locals.get('self', None)), BokehApplication):
                return app
        raise RuntimeError()

    @classmethod
    def get_server(cls) -> Server:
        for frame in inspect.stack():
            if frame.function == 'run_server' and isinstance((server := frame.frame.f_locals.get('server', None)), Server):
                return server
        raise RuntimeError()

    def run_later(self, callback: Callable, *args, **kwargs) -> SessionCallback:
        """
        Run *callback* on next event loop.

        :param callback: callable
        :param args: *callback* arguments
        :param kwargs: *callback* arguments
        """
        return self.document.add_next_tick_callback(functools.partial(callback, *args, **kwargs))

    def run_timeout(self, delay: int, callback: Callable, *args, **kwargs) -> SessionCallback:
        """
        Run *callback* after the  given time.

        :param delay: milliseconds
        :param callback: callable
        :param args: *callback* arguments
        :param kwargs: *callback* arguments
        """
        return self.document.add_timeout_callback(functools.partial(callback, *args, **kwargs), delay)

    def run_periodic(self, cycle: int, callback: Callable, *args, **kwargs) -> SessionCallback:
        """
        Run *callback* on every given time.

        :param cycle: milliseconds
        :param callback: callable
        :param args: *callback* arguments
        :param kwargs: *callback* arguments
        """
        return self.document.add_periodic_callback(functools.partial(callback, *args, **kwargs), cycle)


def run_later(callback: Callable, *args, **kwargs) -> SessionCallback:
    """
    Run *callback* on next event loop.

    :param callback: callable
    :param args: *callback* arguments
    :param kwargs: *callback* arguments
    """
    # TODO bokeh.io.curdoc() ?
    document = BokehApplication.get_application().document
    return document.add_next_tick_callback(functools.partial(callback, *args, **kwargs))


def run_timeout(delay: int, callback: Callable, *args, **kwargs) -> SessionCallback:
    """
    Run *callback* after the  given time.

    :param delay: milliseconds
    :param callback: callable
    :param args: *callback* arguments
    :param kwargs: *callback* arguments
    """
    document = BokehApplication.get_application().document
    return document.add_timeout_callback(functools.partial(callback, *args, **kwargs), delay)


def run_periodic(cycle: int, callback: Callable, *args, **kwargs) -> SessionCallback:
    """
    Run *callback* on every given time.

    :param cycle: milliseconds
    :param callback: callable
    :param args: *callback* arguments
    :param kwargs: *callback* arguments
    """
    document = BokehApplication.get_application().document
    return document.add_periodic_callback(functools.partial(callback, *args, **kwargs), cycle)


def run_server(handlers: BokehApplication | dict[str, BokehApplication], *,
               no_open_browser: bool = False):
    """start bokeh local server and run the application.

    :param handlers: bokeh application, or a dict {path: app}
    :param no_open_browser:
    :return: Never return, except a KeyboardInterrupt is raised
    """
    if isinstance(handlers, BokehApplication):
        handlers = {'/': handlers}

    if '/' not in handlers:
        raise RuntimeError("no '/' handler")

    server = Server({
        p: _setup_application(h)
        for p, h in handlers.items()
    }, num_procs=1)
    server.start()

    if not no_open_browser:
        server.io_loop.add_callback(server.show, "/")

    server.run_until_shutdown()


def _setup_application(handler: BokehApplication):
    def setup(document: Document):
        handler.document = document
        document.title = handler.title

        document.add_root(handler.index())
        document.on_session_destroyed(handler.cleanup)
        document.add_next_tick_callback(handler.start)

    return setup
