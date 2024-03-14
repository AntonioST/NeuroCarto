import dataclasses
import traceback

from bokeh.application.application import SessionContext
from bokeh.document import Document
from bokeh.model import Model
from bokeh.models import PreText

from neurocarto.util.bokeh_app import BokehApplication, run_server
from neurocarto.util.utils import doc_link
from .config import CartoConfig, parse_cli, setup_logger
from .main_app import CartoApp

__all__ = ['CartoAppIndex', 'main']


@doc_link()
class CartoAppIndex(BokehApplication):
    """Support multiple {CartoApp} instance."""

    def __init__(self, config: CartoConfig):
        super().__init__(logger='neurocarto.app.multi')
        self.config = config
        self.sessions: dict[str, int] = {'': 0}

    def setup(self, document: Document):
        session_hash = document.session_context.id
        self.sessions[session_hash] = max(self.sessions.values()) + 1
        session = str(self.sessions[session_hash])

        args = document.session_context.request.arguments
        self.logger.debug('setup[%s] %s', session, args)

        try:
            config = self.new_config(args)
            app = CartoApp(config, logger=f'neurocarto.editor[{session}]')
            app.setup(document)
            document.on_session_destroyed(self.cleanup)
        except BaseException as e:
            self.logger.debug('setup[%s] fail', session, exc_info=e)
            build_error_page(document, 'Error', e)
            self.sessions.pop(session_hash)

    def new_config(self, args: dict[str, list[bytes]]) -> CartoConfig:
        """

        Allow arguments:

        * ``probe``
        * ``selector``
        * ``view`` (multiple times)

        :param args: http request arguments.
        :return: new config
        """
        probe = args.get('probe', [b'npx'])[0].decode()

        try:
            selector = args.get('selector', [])[0].decode()
        except IndexError:
            selector = self.config.selector

        views = list(map(bytes.decode, args.get('view', [])))

        return dataclasses.replace(self.config, probe_family=probe, selector=selector, extra_view=views)

    def index(self) -> Model:
        """Error page"""
        raise RuntimeError()

    def cleanup(self, context: SessionContext):
        session_hash = context.id

        try:
            session = str(self.sessions.pop(session_hash))
        except KeyError:
            pass
        else:
            self.logger.debug('cleanup[%s]', session)


def build_error_page(document: Document, title: str, exc: BaseException):
    """
    generate an error message (python error traceback) page.

    :param document: Bokeh document instance
    :param title: page title
    :param exc: an error
    """
    document.title = title

    content = '\n'.join(traceback.format_exception(exc))
    document.add_root(PreText(text=content))


def main(config: CartoConfig = None):
    """Start application."""
    if config is None:
        config = parse_cli()

    setup_logger(config)

    run_server(CartoAppIndex(config), config)


if __name__ == '__main__':
    main()
