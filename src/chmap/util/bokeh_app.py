import abc
import functools
import logging
from collections.abc import Callable

import bokeh.io
from bokeh.application.application import SessionContext
from bokeh.document import Document
from bokeh.model import Model
from bokeh.plotting import figure as Figure
from bokeh.server.callbacks import SessionCallback
from bokeh.server.server import Server

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


def run_later(callback: Callable, *args, **kwargs) -> SessionCallback:
    """
    Run *callback* on next event loop.

    :param callback: callable
    :param args: *callback* arguments
    :param kwargs: *callback* arguments
    """
    document = bokeh.io.curdoc()
    return document.add_next_tick_callback(functools.partial(callback, *args, **kwargs))


def run_timeout(delay: int, callback: Callable, *args, **kwargs) -> SessionCallback:
    """
    Run *callback* after the given time.

    :param delay: milliseconds
    :param callback: callable
    :param args: *callback* arguments
    :param kwargs: *callback* arguments
    """
    document = bokeh.io.curdoc()
    return document.add_timeout_callback(functools.partial(callback, *args, **kwargs), delay)


def run_periodic(cycle: int, callback: Callable, *args, **kwargs) -> SessionCallback:
    """
    Run *callback* on every given time.

    :param cycle: milliseconds
    :param callback: callable
    :param args: *callback* arguments
    :param kwargs: *callback* arguments
    """
    document = bokeh.io.curdoc()
    return document.add_periodic_callback(functools.partial(callback, *args, **kwargs), cycle)


def run_server(handlers: BokehApplication | dict[str, BokehApplication], *,
               no_open_browser: bool = False):
    """start bokeh local server and run the application.

    :param handlers: bokeh application, or a dict {path: app}
    :param no_open_browser:
    :return: Never return, except a KeyboardInterrupt is raised
    """
    logger = logging.getLogger('chmap.server')

    if isinstance(handlers, BokehApplication):
        handlers = {'/': handlers}

    if '/' not in handlers:
        raise RuntimeError("no '/' handler")

    if logger.isEnabledFor(logging.DEBUG):
        for path, app in handlers.items():
            logger.debug('service map %s -> %s', path, type(app).__name__)

    server = Server({
        p: _setup_application(p, h)
        for p, h in handlers.items()
    }, num_procs=1)

    logger.debug('starting service')
    server.start()
    logger.debug('started service')

    if not no_open_browser:
        logger.debug('open page')
        server.io_loop.add_callback(server.show, "/")

    server.run_until_shutdown()
    logger.debug('stop service')


def _setup_application(path: str, handler: BokehApplication):
    logger = logging.getLogger(f'chmap.server[{path}]')

    def setup(document: Document):
        logger.debug('setup')
        handler.document = document
        document.title = handler.title

        logger.debug('index')
        document.add_root(handler.index())
        document.on_session_destroyed(handler.cleanup)

        logger.debug('start')
        document.add_next_tick_callback(handler.start)

    return setup
