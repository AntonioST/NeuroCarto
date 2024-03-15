from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, cast, TypeVar

from neurocarto.probe import ProbeDesp
from neurocarto.util.utils import import_name, doc_link
from .base import ViewBase, StateView, ControllerView, EditorView, GlobalStateView, ExtensionView

if TYPE_CHECKING:
    from neurocarto.config import CartoConfig
    from neurocarto.main_app import CartoApp

__all__ = ['init_view']

V = TypeVar('V', bound=ViewBase)

SHORT_VIEW = {
    'file': 'neurocarto.views.image:FileImageView',
    'atlas': 'neurocarto.views.atlas:AtlasBrainView',
    'blueprint': 'neurocarto.views.blueprint:BlueprintView',
    'script': 'neurocarto.views.blueprint_script:BlueprintScriptView',
    'history': 'neurocarto.views.record:HistoryView',
}


@doc_link(
    ImageHandler='neurocarto.views.image.ImageHandler',
    ImageView='neurocarto.views.image.ImageView',
    FileImageView='neurocarto.views.image.FileImageView',
    AtlasBrainView='neurocarto.views.atlas.AtlasBrainView',
    BlueprintView='neurocarto.views.blueprint.BlueprintView',
    BlueprintScriptView='neurocarto.views.blueprint_script.BlueprintScriptView',
    HistoryView='neurocarto.views.record.HistoryView'
)
def init_view(config: CartoConfig, probe: ProbeDesp | None, view_type) -> ViewBase | None:
    """

    Recognised type:

    * ``None`` skip
    * {ViewBase} or it subtype

      * It is a {ExtensionView}, also check the *probe* is supported or not.

    * {ImageHandler} or subtype, wrap with {ImageView}. If it is a type, then it should have a no-arg ``__init__``.
    * literal ``'file'`` for {FileImageView}
    * literal ``'atlas'`` for {AtlasBrainView}
    * literal ``'blueprint'`` for {BlueprintView}
    * literal ``'script'`` for {BlueprintScriptView}
    * literal ``'history'`` for {HistoryView}
    * image filepath
    * ``str`` in pattern: ``[PATH:]module.path:attribute`` in type listed above.

    :param config:
    :param probe:
    :param view_type:
    :return:
    """
    from neurocarto.views.image import ImageView, ImageHandler

    try:
        if isinstance(view_type, type) and issubclass(view_type, ViewBase):
            if issubclass(view_type, ExtensionView) and probe is not None and not view_type.is_supported(probe):
                return None

            return view_type(config)

        elif isinstance(view_type, ViewBase):
            return view_type

        elif isinstance(view_type, type) and issubclass(view_type, ImageHandler):
            return ImageView(config, view_type())

        elif isinstance(view_type, ImageHandler):
            return ImageView(config, view_type)

        elif isinstance(view_type, str) and view_type in SHORT_VIEW:
            return import_view(config, probe, SHORT_VIEW[view_type])

        elif isinstance(view_type, str) and is_image(image_file := Path(view_type)):
            from neurocarto.views.image import ImageView, ImageHandler
            return ImageView(config, ImageHandler.from_file(image_file))

        elif isinstance(view_type, str):
            return import_view(config, probe, view_type)
        else:
            raise RuntimeError(f'unknown view_type : {view_type}')

    except BaseException as e:
        logging.getLogger('neurocarto.view').warning('init view fail', exc_info=e)
        pass

    return None


def import_view(config: CartoConfig, probe: ProbeDesp | None, module_path: str) -> ViewBase | None:
    logging.getLogger('neurocarto.view').debug('import %s', module_path)
    return init_view(config, probe, import_name('view base', module_path))


@doc_link()
def install_view(app: CartoApp, view: V) -> V:
    """
    Replace some methods in ViewBase. They are

    * {ViewBase} method {ViewBase#log_message()}
    * {ControllerView} all methods
    * {EditorView} all methods
    * {GlobalStateView} all methods

    :param app: GUI application
    :param view:
    :return: *view* itself
    """

    def log_message(*message, reset=False):
        app.log_message(*message, reset=reset)

    def get_app() -> CartoApp:
        return app

    def get_view(view_type: str | type[ViewBase]) -> ViewBase | None:
        for _view in app.right_panel_views:
            if isinstance(view_type, type) and isinstance(_view, view_type):
                return _view
            elif isinstance(view_type, str) and type(_view).__name__ == view_type:
                return _view
        return None

    def update_probe():
        app.logger.debug('update_probe(%s)', type(view).__name__)
        app.on_probe_update()

    def save_global_state(state=None, *, sync=False, force=False):
        if not getattr(view, 'disable_save_global_state', False) or force:
            app.logger.debug('save_global_state(%s)', type(view).__name__)

            if sync:
                app.save_user_config(direct=False)
            else:
                if state is None:
                    state = cast(StateView, view).save_state()
                app.user_views_config[type(view).__name__] = state
                app.save_user_config(direct=True)

    def restore_global_state(*, reload=False, force=False):
        app.logger.debug('restore_global_state(%s)', type(view).__name__)

        if reload:
            app.load_user_config(reset=False)

        try:
            config = app.user_views_config[type(view).__name__]
        except KeyError:
            if force:
                cast(StateView, view).restore_state({})
        else:
            cast(StateView, view).restore_state(config)

    setattr(view, 'log_message', log_message)

    if isinstance(view, ControllerView):
        setattr(view, 'get_app', get_app)
        setattr(view, 'get_view', get_view)

    if isinstance(view, EditorView):
        setattr(view, 'update_probe', update_probe)

    if isinstance(view, GlobalStateView):
        setattr(view, 'save_global_state', save_global_state)
        setattr(view, 'restore_global_state', restore_global_state)

    return view


def is_image(path: Path) -> bool:
    if not path.is_file():
        return False

    import mimetypes
    mt, _ = mimetypes.guess_type(path)
    return mt is not None and mt.startswith('image/')
