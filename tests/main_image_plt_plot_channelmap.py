from neurocarto.config import parse_cli, CartoConfig
from neurocarto.probe import ProbeDesp, M, E
from neurocarto.probe_npx import ChannelMap
from neurocarto.views.image_plt import PltImageView


class PlotChannelMap(PltImageView):
    def __init__(self, config: CartoConfig):
        super().__init__(config, logger='neurocarto.view.plot_channelmap')

    def on_probe_update(self, probe: ProbeDesp[M, E], chmap: M | None, electrodes: list[E] | None):
        from neurocarto.probe_npx import ChannelMap
        if isinstance(chmap, ChannelMap):
            self.plot_channelmap(chmap)
        else:
            self.set_image(None)

    def plot_channelmap(self, m: ChannelMap):
        self.logger.debug('plot_channelmap')
        from neurocarto.probe_npx import plot

        with self.plot_figure(offset=-80) as ax:
            plot.plot_channelmap_block(ax, chmap=m)
            plot.plot_probe_shape(ax, m, color='k')


if __name__ == '__main__':
    import sys

    from neurocarto.main_app import main

    main(parse_cli([
        *sys.argv[1:],
        '-C', 'res',
        '--debug',
        '--view=-',
        '--view=test:main_image_plt_plot_channelmap:PlotChannelMap',
    ]))
