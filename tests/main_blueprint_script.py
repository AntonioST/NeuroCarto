from neurocarto.main_app import main, CartoApp
from neurocarto.util.bokeh_app import run_later, run_timeout
from neurocarto.views.base import ViewBase, ControllerView
from neurocarto.views.blueprint_script import BlueprintScriptView


class Tester(ViewBase, ControllerView):
    app: CartoApp
    edit: BlueprintScriptView

    @property
    def name(self) -> str:
        return 'Tester'

    def start(self):
        self.app = self.get_app()
        self.edit = self.get_view(BlueprintScriptView)
        if self.app is None or self.edit is None:
            return

        self.set_status('start ...')
        self.edit.visible = True
        self.edit.disable_save_global_state = True
        run_timeout(1000, self.new_probe)

    def new_probe(self):
        self.set_status('create probe ...')
        self.app.on_new(24)
        run_later(self.set_value)

    def set_value(self):
        self.set_status('set value ...')
        self.edit.add_script('demo', 'neurocarto.util.edit._actions:blueprint_simple_init_script_from_activity_data_with_a_threshold')
        self.edit.script_input.value_input = '"res/Fig5d_data.npy", 3500'
        run_later(self.run_script)

    def run_script(self):
        self.set_status('eval content ...')
        self.edit.run_script("demo")
        run_later(self.finish)

    def finish(self):
        self.set_status('finish ...', decay=3)


if __name__ == '__main__':
    import sys
    from neurocarto.config import parse_cli

    main(parse_cli([
        *sys.argv[1:],
        '-C', 'res',
        '--debug',
        '--view=-',
        '--view=blueprint',
        '--view=neurocarto.views.blueprint_script:BlueprintScriptView',
        '--view=tests:main_blueprint_script:Tester',
    ]))
