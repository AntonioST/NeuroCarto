import textwrap

from chmap.main_bokeh import main, ChannelMapEditorApp
from chmap.util.bokeh_app import run_later
from chmap.views.base import ViewBase, ControllerView
from chmap.views.edit_blueprint import InitializeBlueprintView


class Tester(ViewBase, ControllerView):
    app: ChannelMapEditorApp
    edit: InitializeBlueprintView

    @property
    def name(self) -> str:
        return 'Tester'

    def start(self):
        self.app = self.get_app()
        self.edit = self.get_view(InitializeBlueprintView)
        if self.app is None or self.edit is None:
            return

        self.edit.visible = True
        self.edit.disable_save_global_state = True
        run_later(self.new_probe)

    def new_probe(self):
        self.logger.info('new_probe')
        self.app.on_new(24)
        run_later(self.set_content)

    def set_content(self):
        self.logger.info('set_content')
        self.edit.criteria_area.value = textwrap.dedent("""\
        use(NpxProbeDesp)

        alias(F)=FULL
        alias(H)=HALF
        alias(Q)=QUARTER
        alias(L)=LOW
        alias(X)=FORBIDDEN
        
        file=res/Fig5d_data.npy
        print(eval)=f'min={np.nanmin(v)}'
        print(eval)=f'max={np.nanmax(v)}'
        draw()
        X=(s==0)&(y>5000)
        X=(s==1)&(y>5000)
        X=(s==2)&(y>5000)
        X=(s==3)&(y>5000)
        var(active)=(v>=4000)
        var(dca1)=(3500<=y)&(y<=4000)
        var(vca1)=(300<=y)&(y<=1000)
        F=dca1&active
        F=vca1&active
        
        """)
        run_later(self.eval_blueprint)

    def eval_blueprint(self):
        self.logger.info('eval_blueprint')
        self.edit.eval_blueprint()


if __name__ == '__main__':
    import sys
    from chmap.config import parse_cli

    main(parse_cli([
        *sys.argv[1:],
        '-C', 'res',
        '--debug',
        '--view=-',
        '--view=chmap.views.edit_blueprint:InitializeBlueprintView',
        '--view=tests:main_edit_blueprint:Tester',
    ]))
