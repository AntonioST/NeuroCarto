import textwrap

from chmap.main_bokeh import main, ChannelMapEditorApp
from chmap.util.bokeh_app import run_later, run_timeout
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

        self.set_status('start ...')
        self.edit.visible = True
        self.edit.disable_save_global_state = True
        run_timeout(1000, self.new_probe)

    def new_probe(self):
        self.set_status('create probe ...')
        self.app.on_new(24)
        run_later(self.set_content)

    def set_content(self):
        self.set_status('set content ...')
        self.edit.criteria_area.value = textwrap.dedent("""\
        use(NpxProbeDesp)

        alias(F)=FULL
        alias(H)=HALF
        alias(Q)=QUARTER
        alias(L)=LOW
        alias(X)=FORBIDDEN
        
        file=res/Fig5d_data.npy
        >f'min={np.nanmin(v)}, max={np.nanmax(v)}'
        var(v)=bp.interpolate_nan(v)
        draw()
        X=(s==0)&(y>5000)
        X=(s==1)&(y>5000)
        X=(s==2)&(y>5000)
        X=(s==3)&(y>5000)
        var(active)=(v>=3500)
        var(dca1)=(3500<=y)&(y<=4000)
        var(vca1)=(300<=y)&(y<=1000)
        F=dca1&active
        F=vca1&active
        !bp.fill(F, gap=None)
        !bp.fill(F, threshold=10, unset=True)
        !bp.extend(F, 2, threshold=(0, 100))
        !bp.extend(F, 10, H)
        Q=(s==3)&(2000<=y)&(y<=4000)
        L=(s==3)&(1000<=y)&(y<=3000)
        
        """)
        run_later(self.eval_blueprint)

    def eval_blueprint(self):
        self.set_status('eval content ...')
        self.edit.eval_blueprint()
        run_later(self.finish)

    def finish(self):
        self.set_status('finish ...')
        run_timeout(3000, self.set_status, None)


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
