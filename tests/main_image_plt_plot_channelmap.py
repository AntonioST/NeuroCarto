import numpy as np
from numpy._typing import NDArray

from chmap.probe import ProbeDesp, M, E
from chmap.probe_npx import ChannelMap
from chmap.views.image_plt import PltImageHandler, Boundary


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

    def set_image(self, image: NDArray[np.uint] | None,
                  boundary: Boundary = None,
                  offset: float | tuple[float, float] = -80,
                  show_boundary: bool = None):
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
        '--view=test:main_image_plt_plot_channelmap:PlotChannelMap',
    ]))
