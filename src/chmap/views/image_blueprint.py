from chmap.views.image_plt import PltImageHandler

__all__ = ['PlotBlueprint']


class PlotBlueprint(PltImageHandler):
    SUPPORT_TRANSFORM = False
    SUPPORT_ROTATION = False
    SUPPORT_SCALING = False
    SUPPORT_RESOLUTION = False

    def __init__(self):
        super().__init__(logger='chmap.view.plt.plot_blueprint')

    @property
    def name(self) -> str | None:
        return "<b>Blueprint</b>"

    @property
    def description(self) -> str | None:
        return 'plot blueprint beside'

    def on_probe_update(self, probe, chmap, e):
        from chmap.probe_npx import ChannelMap
        if isinstance(chmap, ChannelMap):
            self.plot_channelmap(chmap.probe_type, e)
        else:
            self.set_image(None)

    def plot_channelmap(self, probe_type, e):
        self.logger.debug('plot_channelmap')
        from chmap.probe_npx import plot

        with self.plot_figure(gridspec_kw=dict(top=1, bottom=0, left=0, right=1)) as ax:
            plot.plot_policy_area(ax, probe_type, e, shank_width_scale=0.5)
            plot.plot_probe_shape(ax, probe_type, color=None, label_axis=False, shank_width_scale=0.5)

    def set_image(self, image, boundary=None, offset=-50):
        super().set_image(image, boundary, offset)


if __name__ == '__main__':
    import sys
    from chmap.config import parse_cli
    from chmap.main_bokeh import main

    main(parse_cli([
        *sys.argv[1:],
        '-C', 'res',
        '--debug',
        '--view=-',
        '--view=chmap.views.image_blueprint:PlotBlueprint',
    ]))
