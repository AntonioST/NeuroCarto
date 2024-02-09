from chmap.config import parse_cli, ChannelMapEditorConfig
from chmap.views.image_plt import PltImageView

__all__ = ['BlueprintView']


class BlueprintView(PltImageView):
    def __init__(self, config: ChannelMapEditorConfig):
        super().__init__(config, logger='chmap.view.plot_blueprint')

    @property
    def name(self) -> str | None:
        return "Blueprint"

    @property
    def description(self) -> str | None:
        return 'plot blueprint beside'

    def on_probe_update(self, probe, chmap, electrodes):
        from chmap.probe_npx import ChannelMap
        if not self.visible:
            return

        if isinstance(chmap, ChannelMap):
            self.plot_channelmap(chmap.probe_type, electrodes)
        else:
            self.set_image(None)

    def plot_channelmap(self, probe_type, e):
        self.logger.debug('plot_channelmap')
        from chmap.probe_npx import plot

        with self.plot_figure(gridspec_kw=dict(top=0.99, bottom=0.01, left=0, right=1),
                              offset=-50) as ax:
            plot.plot_category_area(ax, probe_type, e, shank_width_scale=0.5)
            plot.plot_probe_shape(ax, probe_type, color=None, label_axis=False, shank_width_scale=0.5)

            ax.set_xlabel(None)
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_ylabel(None)
            ax.set_yticks([])
            ax.set_yticklabels([])


if __name__ == '__main__':
    import sys

    from chmap.main_bokeh import main

    main(parse_cli([
        *sys.argv[1:],
        '-C', 'res',
        '--debug',
        '--view=-',
        '--view=chmap.views.image_blueprint:BlueprintView',
    ]))
