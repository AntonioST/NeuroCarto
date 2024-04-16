import functools
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from bokeh.application.application import SessionContext
from bokeh.events import MenuItemClick
from bokeh.io import curdoc
from bokeh.models import Div, Select, AutocompleteInput, Toggle, Dropdown, tools, TextAreaInput, UIElement
from bokeh.themes import Theme

from neurocarto.config import CartoConfig, parse_cli, setup_logger
from neurocarto.probe import get_probe_desp, ProbeDesp, M
from neurocarto.util.bokeh_app import BokehApplication, run_server, run_later
from neurocarto.util.bokeh_util import ButtonFactory, col_layout, as_callback, new_help_button
from neurocarto.util.debug import TimeMarker
from neurocarto.util.utils import doc_link
from neurocarto.views import *
from neurocarto.views.record import RecordManager
from . import files

__all__ = ['CartoApp', 'main', 'CartoUserConfig']


@doc_link()
class CartoUserConfig(TypedDict, total=False):
    """
    User config for {CartoApp}.

    All key are optional.

    .. code-block:: json

        {
          "CartoApp": {
            "theme": "",
            "views": [],
            "history": false,
            "overwrite_chmap_file": false,
            "selected_as_pre_selected": false,
          }
        }

    """

    theme: str | dict[str, Any] | None
    """Bokeh theme. see https://docs.bokeh.org/en/latest/docs/reference/themes.html#theme """

    views: list[str]
    """Extra {ViewBase} list"""

    history: bool
    """Enable History. Default False"""

    overwrite_chmap_file: bool
    """overwrite channelmap file by default. Default False"""

    selected_as_pre_selected: bool
    """
    set selected electrode as pre-select category when constructing on blank (or only preselect category) blueprint.
    Default False.
    """


class CartoApp(BokehApplication):
    """Application of neural probe channelmap editor.

    The layout of this application is split into:

    * left: channelmap manipulating button groups.
    * center: figure that contains probe/electrodes, image and curves.
    * right: controls of image, curves that shown in center panel.

    """

    probe: ProbeDesp[M, Any]
    """probe describer"""

    user_views_config: dict[str, Any] = {}
    """user view configuration"""

    right_panel_views_config: dict[str, Any] = {}
    """view configuration, channelmap depended"""

    def __init__(self, config: CartoConfig, *, logger: str = 'neurocarto.editor'):
        super().__init__(logger=logger)
        self.config = config

        self.logger.debug('get get_probe_desp(%s)', config.probe_family)
        self.probe = get_probe_desp(config.probe_family)()
        self.record_manager: RecordManager | None = None
        self.logger.debug('get get_probe_desp() -> %s', type(self.probe).__name__)

        self.load_user_config()
        app_config = self.get_editor_userconfig()
        self._overwrite_channelmap_file = app_config.get('overwrite_chmap_file', False)

    @property
    def title(self) -> str:
        f = self.config.probe_family.upper()
        return f"{f} - NeuroCarto"

    # ==================== #
    # load/save imro files #
    # ==================== #

    @doc_link()
    def load_user_config(self, *, reset=False) -> dict[str, Any]:
        """
        load user config.

        :param reset: reset user config. Otherwise, just update dict.
        :return:
        :see: {files#load_user_config()}
        """
        file = files.user_config_file(self.config)

        try:
            data = files.load_user_config(self.config)
        except FileNotFoundError as e:
            self.logger.debug('user config not found: %s', file, exc_info=e)
            return self.user_views_config
        except IOError as e:
            self.logger.debug('bad user config: %s', file, exc_info=e)
            tmp_file = file.with_name(f'{file.stem}_backup{file.suffix}')
            file.rename(tmp_file)
            self.logger.debug('rename to %s', tmp_file)
            return self.user_views_config
        else:
            self.logger.debug('load user config : %s', file)

        if reset:
            self.user_views_config.update(data)
        else:
            self.user_views_config = data

        return self.user_views_config

    @doc_link()
    def save_user_config(self, direct=False):
        """
        save user config.

        :param direct: Do not update configurations from {GlobalStateView}.
        :see: {files#save_user_config()}
        """
        if not direct:
            for view in self.right_panel_views:
                if isinstance(view, GlobalStateView) and (state := view.save_state(local=False)) is not None:
                    self.logger.debug('on_save() config %s', type(view).__name__)
                    self.user_views_config[type(view).__name__] = state

        file = files.save_user_config(self.config, self.user_views_config)
        self.logger.debug(f'save user config : %s', file)

    def get_editor_userconfig(self) -> CartoUserConfig:
        """
        get editor config from user config.

        :return:
        """
        name = type(self).__name__
        try:
            app_config: CartoUserConfig = self.user_views_config[name]
        except KeyError:
            self.logger.debug('no "%s" in user config, add a default one.', name)
            self.user_views_config[type(self).__name__] = app_config = CartoUserConfig()
            self.save_user_config(direct=True)

        return app_config

    @doc_link()
    def list_chmap_files(self) -> list[Path]:
        """
        List channelmap files under chmap-dir (``-C``, ``--chmap-dir``).

        :return: list of files
        :see: {files#list_channelmap_files()}
        """
        return files.list_channelmap_files(self.config, self.probe)

    def get_chmap_file(self, name: str) -> Path:
        """
        Get channelmap file path.

        :param name: filename. it should be the filename under chmap-dir.
            Otherwise, it needs to contain '/' to load from/save into other place.
        :return: saving path.
        :see: {files#get_channelmap_file()}
        """
        return files.get_channelmap_file(self.config, self.probe, name)

    @doc_link()
    def load_chmap(self, name: str | Path) -> M:
        """
        Load channelmap from file with *name*.

        :param name: filename. See {#get_chmap_file()} for more details.
        :return: channelmap instance.
        """
        if isinstance(name, str):
            file = self.get_chmap_file(name)
        else:
            file = name

        ret = self.probe.load_from_file(file)
        self.log_message(f'load channelmap : {file.name}')
        return ret

    @doc_link()
    def save_chmap(self, name: str, chmap: M) -> Path:
        """
        Save channelmap into file.

        :param name: filename. See {#get_chmap_file()} for more details.
        :param chmap: channelmap instance.
        :return: saved path.
        """
        file = self.get_chmap_file(name)
        self.probe.save_to_file(chmap, file)
        self.log_message(f'save channelmap : {file.name}')
        return file

    @doc_link()
    def get_blueprint_file(self, chmap: str | Path) -> Path:
        """
        Get corresponded blueprint file from channelmap path.

        A blueprint is a numpy array, and saved with a suffix '.blueprint.npy'.
        Sometimes, we encode the channelmap type code inside, e.g. '.CODE.blueprint.npy',
        when the blueprint file does not have the corresponding channelmap file.

        :param chmap: filename. See {#get_chmap_file()} for more details.
        :return: saving path.
        :see: {files#get_blueprint_file()}
        """
        return files.get_blueprint_file(self.config, self.probe, chmap)

    @doc_link()
    def load_blueprint(self, chmap: str | Path) -> bool:
        """
        Load channelmap *chmap* corresponded blueprint.

        :param chmap: filename. See {#get_blueprint_file()} for more details.
        :return: True on success.
        """
        if (electrodes := self.probe_view.electrodes) is None:
            return False

        file = self.get_blueprint_file(chmap)

        try:
            data = np.load(file)
        except FileNotFoundError as e:
            self.log_message(f'File not found : {file}')
            self.logger.warning(f'blueprint file not found : %s', file, exc_info=e)
            return False
        except BaseException as e:
            self.log_message(f'load blueprint fail : {file}')
            self.logger.warning(f'load blueprint file fail : %s', file, exc_info=e)
            return False
        else:
            self.log_message(f'load blueprint : {file.name}')

        self.probe.load_blueprint(data, electrodes)
        return True

    @doc_link()
    def save_blueprint(self, chmap: str | Path) -> Path | None:
        """
        Save channelmap *chmap* corresponded blueprint.

        :param chmap: filename. See {#get_blueprint_file()} for more details.
        :return: saving path. None on failure.
        """
        if (electrodes := self.probe_view.electrodes) is None:
            return None

        file = self.get_blueprint_file(chmap)
        data = self.probe.save_blueprint(electrodes)

        np.save(file, data)
        self.log_message(f'save blueprint : {file.name}')

        return file

    @doc_link()
    def get_view_config_file(self, chmap: str | Path) -> Path:
        """
        Get view components' configurations saving path.

        :param chmap: filename. See {#get_chmap_file()} for more details.
        :return: saving path.
        :see: {files#get_view_config_file()}
        """
        return files.get_view_config_file(self.config, self.probe, chmap)

    @doc_link()
    def load_view_config(self, chmap: str | Path, *, reset=False) -> dict[str, Any]:
        """
        Load view components' configurations.

        :param chmap: filename. See {#get_view_config_file()} for more details.
        :param reset: reset right_view_config or update.
        :return: configuration dict {type_name: Any}.
        """
        file = self.get_view_config_file(chmap)
        if not file.exists():
            self.log_message(f'File not found : {file}')
            return self.right_panel_views_config

        import json
        try:
            with file.open('r') as f:
                data = dict(json.load(f))
        except json.JSONDecodeError as e:
            self.logger.warning('bad view config file %s', file.name, exc_info=e)
            self.log_message(f'bad config : {file.name}')
        else:
            self.log_message(f'load config : {file.name}')

            if reset:
                self.right_panel_views_config.update(data)
            else:
                self.right_panel_views_config = data

        return self.right_panel_views_config

    @doc_link()
    def save_view_config(self, chmap: str | Path, *, direct=False):
        """
        Save view components' configurations.

        :param chmap: filename. See {#get_view_config_file()} for more details.
        :param direct: Do not update configurations from view components.
        """
        if not direct:
            for view in self.right_panel_views:
                if isinstance(view, StateView) and (state := view.save_state()) is not None:
                    self.logger.debug('on_save() config %s', type(view).__name__)
                    self.right_panel_views_config[type(view).__name__] = state

        import json
        file = self.get_view_config_file(chmap)
        with file.open('w') as f:
            json.dump(self.right_panel_views_config, f, indent=2)
            self.log_message(f'save config : {file.name}')

    # ============= #
    # UI components #
    # ============= #

    probe_fig: Figure
    probe_view: ProbeView

    right_panel_views: list[ViewBase]

    input_imro: Select
    output_imro: AutocompleteInput
    message_area: TextAreaInput

    auto_btn: Toggle

    def index(self):
        self.logger.debug('index')
        self.load_user_config(reset=True)

        self.record_manager = None
        if self.get_editor_userconfig().get('history', False):
            self.record_manager = RecordManager(self, self.config)

        # log area goes first
        self.message_area = TextAreaInput(
            rows=10, cols=100, width=290,
            max_length=None, disabled=True
        )

        # middle figure
        self.logger.debug('index figure')
        self.probe_fig = Figure(
            width=600, height=800,
            x_axis_label='(um)', y_axis_label='(um)',
            tools='', toolbar_location='above'
        )
        # figure toolbar
        self.probe_fig.tools.clear()
        self.probe_fig.add_tools(
            (t_drag := tools.PanTool(dimensions='both')),
            tools.WheelPanTool(dimension='height'),
            (t_scroll := tools.WheelZoomTool(dimensions='both')),
            tools.ResetTool(),
        )

        self.probe_fig.toolbar.active_drag = t_drag
        self.probe_fig.toolbar.active_scroll = t_scroll

        # probe
        from neurocarto.views.utils import install_view
        self.probe_view = install_view(self, ProbeView(self.config, self.probe))
        probe_ui = self.probe_view.setup(self.probe_fig)
        if self.record_manager is not None:
            self.record_manager.register(self.probe_view)

        from bokeh.layouts import row, column
        return row(
            column(self._index_left_panel()),
            column(*probe_ui, self.probe_fig),
            column(self._index_right_panel())
        )

    def _set_theme(self, theme: str | Path | dict[str, Any]):
        """
        Json Theme: https://docs.bokeh.org/en/latest/docs/reference/themes.html#theme

        Note: curdoc() work after index()

        :param theme:
        :return:
        """
        document = curdoc()
        if isinstance(theme, str):
            self.logger.debug('set theme %s', theme)
            if theme.endswith('.json'):
                document.theme = Theme(theme)
            else:
                document.theme = theme
        elif isinstance(theme, (Path, dict)):
            document.theme = Theme(theme)
        else:
            raise TypeError()

    def _index_left_panel(self) -> list[UIElement]:
        # 1/1, 1/2, 1/3, ...
        widths = (290, 140, 90, 30)

        self.logger.debug('index left')
        new_btn = ButtonFactory(min_width=widths[1], width_policy='min')

        #
        self.input_imro = Select(
            title='Input file',
            options=[], value="",
            width=widths[0]
        )

        self.output_imro = AutocompleteInput(
            title='Save filename',
            width=widths[0], max_completions=5, case_sensitive=False, restrict=True
        )

        # electrode states buttons
        state_btns = col_layout([
            new_btn(name, functools.partial(self.on_state_change, state=value))
            for name, value in self.probe.possible_states.items()
        ], 2)

        # electrode selecting category buttons
        category_btns = col_layout([
            new_btn(name, functools.partial(self.on_category_change, category=value))
            for name, value in self.probe.possible_categories.items()
        ], 2)

        # channelmap IO buttons
        empty_btn = Dropdown(
            label='New',
            menu=list(self.probe.supported_type),
            min_width=widths[2], align='end', width_policy='min',
            stylesheets=["div.bk-menu { width: 300%; }"]
        )
        empty_btn.on_click(self.on_new)

        load_btn = new_btn('Load', self.on_load, min_width=widths[2], align='end')
        save_btn = new_btn('Save', self.on_save, min_width=widths[2], align='end')

        # electrode selecting helping buttons
        refresh_btn = new_btn('Refresh', self.on_refresh)
        self.auto_btn = Toggle(label='Auto', active=True, min_width=widths[1], width_policy='min')
        self.auto_btn.on_change('active', as_callback(self.on_autoupdate))

        from bokeh.layouts import row
        return [
            # Channelmap IO
            Div(text="<b>ChannelMap File</b>"),
            self.input_imro,
            row(empty_btn, load_btn, save_btn),
            self.output_imro,
            # Electrode Select
            row(Div(text="<b>Electrode State</b>"),
                new_help_button('Manually select electrodes directly', position='right')),
            *state_btns,
            # Electrode Category
            row(Div(text="<b>Electrode Category</b>"),
                new_help_button('Set electrode category for programming electrode selection', position='right')),
            *category_btns,
            row(self.auto_btn, refresh_btn),
            # log message
            Div(text="<b>Log</b>"),
            self.message_area,
            new_btn('clear', self.log_clear, min_width=widths[3])
        ]

    def _index_right_panel(self) -> list[UIElement]:
        from neurocarto.views.utils import install_view

        self.logger.debug('index right')
        self.right_panel_views = []

        for view_type in self.install_right_panel_views(self.config):
            if (view := init_view(self.config, self.probe, view_type)) is not None:
                self.right_panel_views.append(install_view(self, view))
                if self.record_manager is not None and isinstance(view, RecordView):
                    self.record_manager.register(view)

        uis = []
        for view in self.right_panel_views:
            uis.extend(view.setup(self.probe_fig))

        return uis

    @doc_link()
    def install_right_panel_views(self, config: CartoConfig) -> list:
        """
        :param config:
        :return: list of item that recognised by {init_view()}
        """
        ret: list = self.probe.extra_controls(config)

        ext = []

        ext.extend(self.get_editor_userconfig().get('views', [
            'neurocarto.views.data_density:ElectrodeDensityDataView',
            'neurocarto.views.view_efficient:ElectrodeEfficiencyData',
            'blueprint', 'atlas'
        ]))
        ext.extend(config.extra_view)

        if '-' in ext:
            ext = ext[ext.index('-') + 1:]

        ret.extend(ext)
        return ret

    def start(self):
        self.logger.debug('start')

        if (theme := self.get_editor_userconfig().get('theme', None)) is not None:
            try:
                self._set_theme(theme)
            except BaseException as e:
                self.logger.warning('set theme', exc_info=e)

        self.reload_input_imro_list()

        self.probe_view.start()

        for view in self.right_panel_views:
            self.logger.debug('start %s', type(view).__name__)
            view.start()

        if (open_file := self.config.open_file) is not None:
            run_later(self.load_file, open_file)

    def cleanup(self, context: SessionContext):
        super().cleanup(context)
        self.save_user_config()

    # ========= #
    # callbacks #
    # ========= #

    def on_new(self, e: int | str | MenuItemClick):
        if isinstance(e, MenuItemClick):
            if (item := e.item) is None:
                self.logger.debug('on_new()')
                return
        else:
            item = e

        if isinstance(item, str):
            probe_type = self.probe.supported_type[item]
            self.logger.debug('on_new(%d)=%s', probe_type, item)
            self.log_message(f'new probe({item})')
        elif isinstance(item, int):
            probe_type = item
            self.logger.debug('on_new(%d)', probe_type)
            self.log_message(f'new probe[{probe_type}]')
        else:
            raise TypeError()

        self.probe_view.reset(probe_type)

        if len(self.output_imro.value_input) == 0:
            self.output_imro.value_input = "New"

        self.on_probe_update()

    def on_load(self):
        if len(name := self.input_imro.value) == 0:
            self.logger.debug('on_load()')
            return

        self.load_file(name)

    def load_file(self, name: str | Path):
        self.logger.debug('on_load(%s)', name)

        file = name

        try:
            if isinstance(name, str):
                file = self.get_chmap_file(name)
                chmap = self.load_chmap(name)
            else:
                chmap = self.load_chmap(file)
        except FileNotFoundError as x:
            self.log_message(f'File not found : {file}')
            self.logger.warning(f'channelmap file not found : %s', file, exc_info=x)
            return

        self.output_imro.value_input = self._load_file_save_name(file)

        self.probe_view.reset(chmap)
        self.load_blueprint(file)

        config = self.load_view_config(file, reset=True)
        for view in self.right_panel_views:
            if isinstance(view, StateView):
                try:
                    _config = config[type(view).__name__]
                except KeyError:
                    pass
                else:
                    self.logger.debug('on_load() config %s', type(view).__name__)
                    try:
                        view.restore_state(_config)
                    except RuntimeError as e:
                        self.log_message(type(view).__name__ + '::' + repr(e))

        self.on_probe_update()

    def _load_file_save_name(self, file: Path) -> str:
        name = file.stem.replace('.', '_')
        if self._overwrite_channelmap_file:
            return name

        ext = self.probe.channelmap_file_suffix[0]

        i = 0
        while (ret := file.with_name(f'{name}_{i}{ext}')).exists():
            i += 1

        return ret.stem

    def reload_input_imro_list(self, preselect: str = None):
        if preselect is None:
            preselect = self.input_imro.value

        file_list = self.list_chmap_files()
        if len(self.probe.channelmap_file_suffix) == 1:
            opt_list = [(str(f), f.stem) for f in file_list]
        else:
            opt_list = [(str(f), f.name) for f in file_list]

        self.input_imro.options = [""] + opt_list

        name_list = [it[1] for it in opt_list]

        if preselect in name_list:
            self.input_imro.value = preselect
        else:
            self.input_imro.value = ""

        self.output_imro.completions = name_list

    def on_save(self):
        name = self.output_imro.value_input
        if len(name) == 0:
            self.logger.debug('on_save()')
            self.log_message('empty output filename')
            return

        self.logger.debug('on_save(%s)', name)
        chmap = self.probe_view.channelmap
        if chmap is None:
            self.log_message(f'no channelmap')

        elif not self.probe.is_valid(chmap):
            self.log_message(f'incomplete channelmap')
            path = self.get_blueprint_file(name)
            if (code := self.probe.channelmap_code(chmap)) is not None:
                path = path.with_name(path.name.replace('.blueprint', f'.{code}.blueprint'))
            self.save_blueprint(path)

        else:
            path = self.save_chmap(name, chmap)
            self.save_blueprint(path)
            self.save_view_config(path)

            self.output_imro.value_input = path.stem
            self.reload_input_imro_list(path.stem)

    def on_state_change(self, state: int):
        desp = self.probe.state_description(state)

        if desp is not None:
            self.logger.debug('on_state_change(%d)=%s', state, desp)
        else:
            self.logger.debug('on_state_change(%d)', state)

        try:
            self.probe_view.set_state_for_captured(state)
        except BaseException:
            self.log_message(f'set state {desp} fail')
        else:
            self.on_probe_update()

    def on_category_change(self, category: int):
        desp = self.probe.category_description(category)

        if desp is not None:
            self.logger.debug('on_category_change(%d)=%s', category, desp)
        else:
            self.logger.debug('on_category_change(%d)', category)

        electrodes = None
        trigger_fresh = True
        if self.get_editor_userconfig().get('selected_as_pre_selected', False):
            if len([it for it in self.probe_view.electrodes if it.category not in (ProbeDesp.CATE_UNSET, ProbeDesp.CATE_SET)]) == 0:
                electrodes = self.probe_view.get_captured_electrodes_index(self.probe_view.data_electrodes[ProbeDesp.STATE_USED], reset=False)
                trigger_fresh = False

        try:
            self.probe_view.set_category_for_captured(category, electrodes=electrodes)
        except BaseException as e:
            self.log_message(f'set category {desp} fail')
            self.logger.warning('set category %s fail', desp, exc_info=e)
            return

        if self.auto_btn.active and trigger_fresh:
            self.on_refresh()
        else:
            self.on_probe_update()

    def on_probe_update(self):
        self.probe_view.update_electrode()

        mark = TimeMarker()
        for view in self.right_panel_views:
            if isinstance(view, DynamicView):
                view_class = type(view).__name__

                mark.reset()
                view.on_probe_update(self.probe, self.probe_view.channelmap, self.probe_view.electrodes)
                t = mark()

                self.logger.debug('on_probe_update(%s) used %.2f sec', view_class, t)

    def on_autoupdate(self, active: bool):
        self.logger.debug('on_autoupdate(active=%s)', active)
        if active:
            self.on_refresh()

    def on_refresh(self):
        try:
            self.probe_view.refresh_selection()
        except BaseException:
            self.log_message('refresh fail')
        else:
            self.on_probe_update()

    def log_message(self, *message, reset=False):
        """
        print messages at log area.

        :param message: messages.
        :param reset: reset text area.
        """
        message = '\n'.join(message)
        self.logger.info(message)

        if reset:
            self.message_area.value = message
        else:
            text = self.message_area.value
            self.message_area.value = message + '\n' + text

    def log_clear(self):
        """
        Clear log text area.
        """
        self.message_area.value = ""


def main(config: CartoConfig = None):
    """Start application."""
    if config is None:
        config = parse_cli()

    setup_logger(config)

    run_server(CartoApp(config), config)


if __name__ == '__main__':
    main()
