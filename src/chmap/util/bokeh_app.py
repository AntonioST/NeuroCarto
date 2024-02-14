from __future__ import annotations

import abc
import functools
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import bokeh.io
from bokeh.document import Document
from bokeh.model import Model
from bokeh.plotting import figure as Figure
from bokeh.server.server import Server

from chmap.util.utils import doc_link

if TYPE_CHECKING:
    from bokeh.application.application import SessionContext
    from bokeh.server.callbacks import PeriodicCallback, TimeoutCallback, NextTickCallback

    from chmap.config import ChannelMapEditorConfig

__all__ = [
    'BokehApplication',
    'run_later',
    'run_timeout',
    'run_periodic',
    'run_server',
    'remove_timeout',
    'remove_periodic',
    'Figure'
]


class BokehApplication(metaclass=abc.ABCMeta):
    """Bokeh Application of a single page"""

    logger: logging.Logger

    def __init__(self, *, logger: str | logging.Logger = None):
        if isinstance(logger, str):
            self.logger = logging.getLogger(logger)
        elif isinstance(logger, logging.Logger):
            self.logger = logger
        else:
            self.logger = logging.getLogger(f'chmap.app.{type(self).__name__}')

        self.logger.debug('init()')

    @property
    def title(self) -> str:
        """Application title (shown in web page title)"""
        return type(self).__name__

    def setup(self, document: Document):
        self.logger.debug('setup')
        document.title = self.title
        document.add_root(self.index())
        document.on_session_destroyed(self.cleanup)
        document.add_next_tick_callback(self.start)

    @abc.abstractmethod
    def index(self) -> Model:
        """Web-page document content"""
        pass

    def start(self):
        """Invoked when session set"""
        self.logger.debug('start()')

    def cleanup(self, context: SessionContext):
        """Invoked when session destroyed"""
        self.logger.debug('cleanup()')


def run_later(callback: Callable, *args, **kwargs) -> NextTickCallback:
    """
    Run *callback* on next event loop.

    :param callback: callable
    :param args: *callback* arguments
    :param kwargs: *callback* arguments
    """
    document = bokeh.io.curdoc()
    return document.add_next_tick_callback(functools.partial(callback, *args, **kwargs))


def run_timeout(delay: int, callback: Callable, *args, **kwargs) -> TimeoutCallback:
    """
    Run *callback* after the given time.

    :param delay: milliseconds
    :param callback: callable
    :param args: *callback* arguments
    :param kwargs: *callback* arguments
    """
    document = bokeh.io.curdoc()
    return document.add_timeout_callback(functools.partial(callback, *args, **kwargs), delay)


def remove_timeout(callback: TimeoutCallback):
    document = bokeh.io.curdoc()
    document.remove_timeout_callback(callback)


def run_periodic(cycle: int, callback: Callable, *args, **kwargs) -> PeriodicCallback:
    """
    Run *callback* on every given time.

    :param cycle: milliseconds
    :param callback: callable
    :param args: *callback* arguments
    :param kwargs: *callback* arguments
    """
    document = bokeh.io.curdoc()
    return document.add_periodic_callback(functools.partial(callback, *args, **kwargs), cycle)


def remove_periodic(callback: PeriodicCallback):
    document = bokeh.io.curdoc()
    document.remove_periodic_callback(callback)


@doc_link()
def run_server(handlers: BokehApplication | dict[str, BokehApplication],
               config: ChannelMapEditorConfig):
    """start bokeh local server and run the application.

    :param handlers: bokeh application, or a dict {path: app}
    :param config:
    :return: Never return, except a {KeyboardInterrupt} is raised
    """
    logger = logging.getLogger('chmap.server')

    if isinstance(handlers, BokehApplication):
        handlers = {'/': handlers}

    if '/' not in handlers:
        raise RuntimeError("no '/' handler")

    if logger.isEnabledFor(logging.DEBUG):
        for path, app in handlers.items():
            logger.debug('service map %s -> %s', path, type(app).__name__)

    if (port := config.server_port) is None:
        from bokeh.resources import DEFAULT_SERVER_PORT
        port = DEFAULT_SERVER_PORT

    server = Server({
        p: h.setup
        for p, h in handlers.items()
    }, address=config.server_address, port=port, num_procs=1)

    logger.debug('starting service')
    server.start()
    logger.debug('started service')

    if not config.no_open_browser:
        logger.debug('open page')
        server.io_loop.add_callback(server.show, "/")

    server.run_until_shutdown()
    logger.debug('stop service')
