from chmap.probe import ProbeDesp, M, E
from chmap.probe_npx import ChannelMap
from chmap.views.image_plt import PltImageHandler


class PlotChannelMap(PltImageHandler):

    def __init__(self):
        super().__init__(logger='chmap.view.plt.plot_channel')

    def on_probe_update(self, probe: ProbeDesp[M, E], chmap: M | None, e: list[E] | None):
        from chmap.probe_npx import ChannelMap
        if isinstance(chmap, ChannelMap):
            self.plot_channelmap(chmap)
        else:
            self.set_image(None)

    def plot_channelmap(self, m: ChannelMap):
        self.logger.debug('plot_channelmap')
        from chmap.probe_npx import plot

        with self.plot_figure() as ax:
            plot.plot_channelmap_block(ax, chmap=m)
            plot.plot_probe_shape(ax, m, color='k')


if __name__ == '__main__':
    import sys
    from chmap.config import parse_cli
    from chmap.main_bokeh import main

    main(parse_cli([
        *sys.argv[1:],
        '-C', 'res',
        '--debug',
        '--view=-',
        '--view=test:main_image_plt_plot_channelmap:PlotChannelMap',
    ]))
