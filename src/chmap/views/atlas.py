from __future__ import annotations

from typing import get_args, TypedDict, Final, NamedTuple

import numpy as np
from bokeh.events import DoubleTap
from bokeh.models import ColumnDataSource, GlyphRenderer, Select, Slider, UIElement, MultiChoice, Div, CheckboxGroup, tools, Range
from numpy.typing import NDArray

from chmap.config import ChannelMapEditorConfig
from chmap.util.atlas_brain import BrainGlobeAtlas, get_atlas_brain, REFERENCE
from chmap.util.atlas_slice import SlicePlane, SLICE, SliceView
from chmap.util.atlas_struct import Structures
from chmap.util.bokeh_util import ButtonFactory, SliderFactory, as_callback, is_recursive_called, new_help_button
from chmap.util.util_numpy import closest_point_index
from chmap.views.base import Figure, StateView, BoundView, BoundaryState

__all__ = ['AtlasBrainView', 'AtlasBrainViewState', 'Label']

LABEL_REFS = ['probe', 'image', 'bregma']


class Label(NamedTuple):
    text: str
    pos: tuple[float, float, float]
    """position, either (x, y, 1) or (ap, dv, ml) depends on origin"""
    ref: int
    color: str

    @property
    def origin(self) -> str:
        return LABEL_REFS[self.ref]


class AtlasBrainViewState(TypedDict, total=False):
    atlas_brain: str
    brain_slice: SLICE | None
    slice_plane: int | None
    slice_rot_w: int
    slice_rot_h: int
    image_dx: float
    image_dy: float
    image_sx: float
    image_sy: float
    image_rt: float
    regions: list[str]


class AtlasBrainView(BoundView, StateView[AtlasBrainViewState]):
    """
    Atlas mouse brain displaying view component.

    Used command-line arguments:
    * '--atlas' : brain name.
    * '--atlas-root' : data saving directory.

    Event Call chain
    ----------------

    ::

        (boundary UI components)
            -/-> (event callback)
                -> update_boundary_transform()
                    -> on_boundary_transform()

        (atlas UI components)
            (slice_select) -/-> (event callback)
                -> update_brain_view()
                    -> update_brain_slice(update_image=False)
                    -> update_boundary_transform()
                        -> on_boundary_transform()
                            -> update_image()
                                -> update_region_image()
            (region_choose) -/-> (event callback)
                -> update_region_image()
            (other) -/-> (event callback)
                -> update_brain_slice()
                    -> update_image()
                        -> update_region_image()

    """

    brain: Final[BrainGlobeAtlas]

    data_brain: ColumnDataSource
    render_brain: GlyphRenderer

    data_region: ColumnDataSource
    render_region: GlyphRenderer

    data_labels: ColumnDataSource
    render_labels: GlyphRenderer

    def __init__(self, config: ChannelMapEditorConfig, *, logger: str = 'chmap.view.atlas'):
        super().__init__(config, logger=logger)

        self.logger.debug('init(%s)', config.atlas_name)
        self.brain = get_atlas_brain(config.atlas_name, config.atlas_root)
        self._structure = Structures.of(self.brain)
        self._labels: list[Label] = []

        self.data_brain = ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))
        self.data_region = ColumnDataSource(data=dict(image=[], x=[], y=[], dw=[], dh=[]))
        self.data_labels = ColumnDataSource(data=dict(i=[], x=[], y=[], label=[], color=[], ap=[], dv=[], ml=[]))

        self._brain_view: SliceView | None = None
        self._brain_slice: SlicePlane | None = None
        self._regions: dict[str, int] = {}

    @property
    def name(self) -> str:
        return 'Atlas brain'

    # ========== #
    # properties #
    # ========== #

    @property
    def width(self) -> float:
        try:
            return self._brain_slice.width
        except (TypeError, AttributeError):
            return 0

    @property
    def height(self) -> float:
        try:
            return self._brain_slice.height
        except (TypeError, AttributeError):
            return 0

    @property
    def brain_view(self) -> SliceView:
        assert self._brain_view is not None
        return self._brain_view

    @property
    def brain_slice(self) -> SlicePlane | None:
        return self._brain_slice

    # ============= #
    # UI components #
    # ============= #

    slice_select: Select
    plane_slider: Slider
    rotate_hor_slider: Slider
    rotate_ver_slider: Slider
    region_choose: MultiChoice

    checkbox_group: CheckboxGroup
    checkbox_groups: dict[str, UIElement]

    _figure_x_range: Range = None
    _figure_y_range: Range = None

    def _setup_render(self, f: Figure,
                      palette: str = 'Greys256',
                      palette_region: str = 'Turbo256',
                      boundary_color: str = 'black',
                      **kwargs):
        self._figure_x_range = f.x_range
        self._figure_y_range = f.y_range

        self.render_brain = f.image(
            'image', x='x', y='y', dw='dw', dh='dh', source=self.data_brain,
            palette=palette, level="image", global_alpha=0.5, syncable=False,
        )

        self.render_region = f.image(
            'image', x='x', y='y', dw='dw', dh='dh', source=self.data_region,
            palette=palette_region, level="image", global_alpha=0.3, syncable=False,
        )

        self.render_labels = f.scatter(
            x='x', y='y', color='color', source=self.data_labels, size=10,
        )
        f.on_event(DoubleTap, self._on_label_tap)

        self.setup_boundary(f, boundary_color=boundary_color, boundary_desp='drag atlas brain image')

        # toolbar
        f.tools.append(tools.HoverTool(
            description='Atlas labels',
            renderers=[self.render_labels],
            tooltips=[
                ("bregma (ap,dv,ml) mm", "(@ap, @dv, @ml)"),
                ("probe (x,y) um", "(@x, @y)"),
                ('Label', '[@i] @label'),
            ]
        ))

    def _setup_title(self, **kwargs) -> list[UIElement]:
        ret = super()._setup_title(**kwargs)

        self.checkbox_group = CheckboxGroup(labels=['Shear', 'Rotation', 'Scaling', 'Masking'], inline=True)
        self.checkbox_group.on_change('active', as_callback(self._on_checkbox_active))
        ret.insert(-1, self.checkbox_group)

        return ret

    def _setup_content(self, slider_width: int = 300,
                       rotate_steps=(-1000, 1000, 5),
                       **kwargs) -> list[UIElement]:
        from bokeh.layouts import row, column

        new_btn = ButtonFactory(min_width=100, min_height=30, width_policy='min', height_policy='min')
        new_slider = SliderFactory(width=slider_width, align='end')
        self.checkbox_groups = {}

        # slicing
        slice_view_options = list(get_args(SLICE))
        self.slice_select = Select(
            value=slice_view_options[0],
            options=slice_view_options,
            width=100,
        )
        self.slice_select.on_change('value', as_callback(self._on_slice_selected))
        self.plane_slider = new_slider('Slice Plane (um)', (0, 1, 1, 0), self._on_slice_changed)

        # shear
        self.rotate_hor_slider = new_slider('horizontal rotation (um)', rotate_steps, self._on_rotate_changed)
        self.rotate_ver_slider = new_slider('vertical rotation (um)', rotate_steps, self._on_rotate_changed)

        reset_rth = new_btn('reset', self._on_reset_rotate_horizontal)
        reset_rtv = new_btn('reset', self._on_reset_rotate_vertical)

        self.checkbox_groups['Shear'] = col_shear = column(
            row(reset_rth, self.rotate_hor_slider),
            row(reset_rtv, self.rotate_ver_slider),
        )

        # rotation
        self.checkbox_groups['Rotation'] = col_rot = row(*self.setup_rotate_slider(new_btn=new_btn, new_slider=new_slider))

        # scaling
        self.checkbox_groups['Scaling'] = col_scale = row(*self.setup_scale_slider(new_btn=new_btn, new_slider=new_slider))

        # masking
        self.region_choose = MultiChoice(
            options=[
                ((structure := self._structure[acronym]).acronym, f'{acronym} ({structure.name})')
                for acronym in sorted(self._structure.regions)
            ],
            width=350,
        )
        self.region_choose.on_change('value', as_callback(self._on_region_choose))
        mask_help = new_help_button('type region names to color label the corresponding areas.')
        self.checkbox_groups['Masking'] = col_mask = row(Div(text='mask'), self.region_choose, mask_help)

        return [
            row(self.slice_select, self.plane_slider),
            col_shear,
            col_rot,
            col_scale,
            col_mask,
        ]

    def _on_slice_selected(self, s: str):
        if is_recursive_called():
            return

        self.update_brain_view(s)

    def _on_slice_changed(self, s: int):
        if is_recursive_called():
            return

        view = self.brain_view
        if view.name == 'sagittal':
            d = view.n_plane * view.resolution / 2
            p = int((s + d) / view.resolution)
        else:
            p = int(s / view.resolution)

        self.update_brain_slice(p)

    def _on_checkbox_active(self, active: list[int]):
        for i, n in enumerate(self.checkbox_group.labels):
            if (ui := self.checkbox_groups.get(n, None)) is not None:
                ui.visible = i in active

    def _on_rotate_changed(self):
        if is_recursive_called():
            return

        if (p := self._brain_slice) is not None:
            r = p.slice.resolution
            x = int(self.rotate_hor_slider.value / r)
            y = int(self.rotate_ver_slider.value / r)
            q = p.with_offset(x, y)
            self.update_brain_slice(q)

    def _on_reset_rotate_horizontal(self):
        try:
            self.rotate_hor_slider.value = 0
        except AttributeError:
            pass

    def _on_reset_rotate_vertical(self):
        try:
            self.rotate_ver_slider.value = 0
        except AttributeError:
            pass

    def _on_region_choose(self, value: list[str]):
        self.logger.debug('choose(%s)', value)

        self._regions = {
            region: i + 1
            for i, region in enumerate(value)
        }

        self.update_region_image()

    def _on_label_tap(self, tap: DoubleTap):
        if (label := self.find_label((tap.x, tap.y))) is not None:
            if label.origin == 'bregma':
                origin = REFERENCE['bregma'][self.brain_view.brain.atlas_name]
                ap, dv, ml = label.pos
                ap = origin[0] - ap * 1000
                dv = origin[1] + dv * 1000
                ml = origin[2] + ml * 1000
                p, _, _ = self.brain_view.project((ap, dv, ml), um=True)
                self.update_brain_slice(p)

    # ========= #
    # load/save #
    # ========= #

    def save_state(self) -> AtlasBrainViewState:
        boundary = self.get_boundary_state()

        self.logger.debug('save()')
        return AtlasBrainViewState(
            atlas_brain=self.brain.atlas_name,
            brain_slice=None if (p := self._brain_view) is None else p.name,
            slice_plane=None if (p := self._brain_slice) is None else p.plane,
            slice_rot_w=0 if p is None else p.dw,
            slice_rot_h=0 if p is None else p.dh,
            image_dx=boundary['dx'],
            image_dy=boundary['dy'],
            image_sx=boundary['sx'],
            image_sy=boundary['sy'],
            image_rt=boundary['rt'],
            regions=list(self._regions)
        )

    def restore_state(self, state: AtlasBrainViewState):
        if self.brain.atlas_name != state['atlas_brain']:
            raise RuntimeError()

        self.logger.debug('restore()')
        self.update_brain_view(state['brain_slice'])

        dp = state['slice_plane']
        dw = state['slice_rot_w']
        dh = state['slice_rot_h']
        brain_slice = self.brain_view.plane_at(dp).with_offset(dw, dh)
        self.update_brain_slice(brain_slice, update_image=False)

        self.update_boundary_transform(p=(state['image_dx'], state['image_dy']), s=(state['image_sx'], state['image_sx']), rt=state['image_rt'])

        try:
            self.region_choose.value = state['regions']
        except (AttributeError, KeyError):
            pass

    # ================ #
    # updating methods #
    # ================ #

    def start(self):
        self._on_checkbox_active([])

        if self._brain_view is None:
            try:
                view = self.slice_select.value
            except AttributeError:
                view = 'coronal'

            self.update_brain_view(view)
        elif (plane := self._brain_slice) is not None:
            self.update_image(plane.image)

    # ====== #
    # Labels #
    # ====== #

    def clear_labels(self):
        """Clear all labels"""
        self._labels = []
        self.data_labels.data = dict(i=[], x=[], y=[], label=[], color=[], ap=[], dv=[], ml=[])

    def len_label(self) -> int:
        """number of the labels"""
        return len(self._labels)

    def get_label(self, i: int) -> str:
        """
        Get the label text at index *i*.

        :param i: index
        :return: label text
        :raises IndexError: index *i* out of bound
        """
        return self._labels[i].text

    def find_label(self, pos: tuple[float, float]) -> Label | None:
        """
        Find a label around the given position.

        :param pos: (x, y) um
        :return: found label
        """
        data = self.data_labels.data
        if len(x := data['x']) == 0:
            return

        dx = (self._figure_x_range.end - self._figure_x_range.start) / 30
        dy = (self._figure_y_range.end - self._figure_y_range.start) / 30
        th = min(dx, dy)

        labels = np.vstack([x, data['y']]).T
        if (i := closest_point_index(labels, pos, th)) is not None:
            return self._labels[i]
        return None

    def index_label(self, text: str) -> int | None:
        """
        Find index of a label which its content equals to *text*.

        :param text: label text
        :return: label index. ``None`` if not found.
        """
        for i, label in enumerate(self._labels):
            if label.text == text:
                return i
        return None

    def add_label(self, text: str,
                  pos: tuple[float, float] | tuple[float, float, float], *,
                  origin: str = 'bregma',
                  color: str = 'cyan',
                  replace=True):
        """
        Add a label.

        :param text: label text
        :param pos: label position
        :param origin: origin reference point
        :param color: label color
        :param replace: replace label which has same text content
        """
        try:
            ref = LABEL_REFS.index(origin)
        except ValueError:
            ref = None

        if ref is None:
            try:
                REFERENCE[origin][self.brain_view.brain.atlas_name]
            except KeyError as e:
                raise ValueError(f'unknown origin type : {origin}') from e

            ref = len(LABEL_REFS)
            LABEL_REFS.append(origin)

        i: int | None = None
        if replace:
            i = self.index_label(text)

        if len(pos) == 2:
            pos = (*pos, 1.0)

        label = Label(text, pos, ref, color)
        self._labels.append(label)

        if i is None:
            self.data_labels.stream(self._transform_labels([label], start=len(self._labels)))
        else:
            self.del_label(i)

    def del_label(self, index: int | list[int]):
        """
        Remove labels.

        :param index: index, list of index.
        """
        if isinstance(index, int):
            index = [index]

        if len(index) == 0:
            return

        index = set(index)
        self._labels = [it for i, it in enumerate(self._labels) if i not in index]
        self.update_label_position()

    def update_label_position(self):
        self.data_labels.data = self._transform_labels(self._labels)

    def _transform_labels(self, labels: list[Label], start: int = 0) -> dict:
        if (n := len(labels)) == 0:
            return dict(i=[], x=[], y=[], label=[], color=[], ap=[], dv=[], ml=[])

        ii = list(range(start, start + n))
        o = np.array([it.ref for it in labels])  # Array[ref:int, N]
        p = np.array([it.pos for it in labels]).T  # Array[float, 3, N]
        t = [it.text for it in labels]
        c = [it.color for it in labels]

        a, a_ = self._prepare_affine_matrix()

        origin = REFERENCE['bregma'][self.brain_view.brain.atlas_name]

        qp = np.zeros_like(p)  # Array[float, 3, N]
        qb = np.zeros((3, n), dtype=float)  # Array[float, 3, N]
        for i, ref in enumerate(LABEL_REFS):
            if not np.any(mask := o == i):
                continue

            match ref:
                case 'probe':
                    q = p[:, mask]  # Array[float, 3, N']
                    qp[:, mask] = q
                    qb[:, mask] = self._image2bregma(a_ @ q, origin)

                case 'image':
                    q = p[:, mask]  # Array[float, 3, N']
                    qp[:, mask] = a @ q
                    qb[:, mask] = self._image2bregma(q, origin)

                case 'bregma':
                    q = p[:, mask]  # Array[float, (ap,dv,ml), N']
                    qb[:, mask] = q
                    qp[:, mask] = a @ self._bregma2image(q, origin)

                case str(bregma):
                    q = p[:, mask]  # Array[float, (ap,dv,ml), N']
                    qb[:, mask] = q
                    qp[:, mask] = a @ self._bregma2image(q, REFERENCE[bregma][self.brain_view.brain.atlas_name])

        return dict(i=ii, x=qp[0], y=qp[1], label=t, color=c, ap=qb[0], dv=qb[1], ml=qb[2])

    def _prepare_affine_matrix(self) -> tuple[NDArray[np.float_], NDArray[np.float_]]:
        """

        :return: tuple of (A, A^-1)
        """
        boundary = self.get_boundary_state()
        dx = boundary['dx']
        dy = boundary['dy']
        sx = boundary['sx']
        sy = boundary['sy']
        rt = np.deg2rad(boundary['rt'])

        td = np.array([
            [1, 0, dx],
            [0, 1, dy],
            [0, 0, 1],
        ], dtype=float)
        td_ = np.array([
            [1, 0, -dx],
            [0, 1, -dy],
            [0, 0, 1],
        ], dtype=float)
        cos = np.cos(rt)
        sin = np.sin(rt)
        tr = np.array([
            [cos, -sin, 0],
            [sin, cos, 0],
            [0, 0, 1],
        ], dtype=float)
        tr_ = np.array([
            [cos, sin, 0],
            [-sin, cos, 0],
            [0, 0, 1],
        ], dtype=float)
        ts = np.array([
            [sx, 0, 0],
            [0, sy, 0],
            [0, 0, 1],
        ], dtype=float)
        ts_ = np.array([
            [1 / sx, 0, 0],
            [0, 1 / sy, 0],
            [0, 0, 1],
        ], dtype=float)
        return td @ ts @ tr, tr_ @ ts_ @ td_

    def _image2bregma(self, q: NDArray[np.float_], origin: tuple[float, float, float]) -> NDArray[np.float_]:
        """

        :param q: Array[um:float, (x, y, 1), N]
        :param origin: (ap, dv, ml) in um
        :return: Array[mm:float, (ap, dv, ml), N]
        """
        cx = self.width / 2
        cy = self.height / 2
        q[0] = q[0] + cx
        q[1] = cy - q[1]
        q = self.brain_slice.coor_on(q[(0, 1), :].T, um=True).T  # Array[int, (ap,dv,ml), N]
        q = q * self.brain_view.resolution  # Array[um:float, (ap,dv,ml), N]
        q[0] = origin[0] - q[0]
        q[1] = q[1] - origin[1]
        q[2] = q[2] - origin[2]
        return q / 1000  # Array[mm:float, (ap,dv,ml), N]

    def _bregma2image(self, q: NDArray[np.float_], origin: tuple[float, float, float]) -> NDArray[np.float_]:
        """

        :param q: Array[mm:float, (ap, dv, ml), N]
        :param origin: (ap, dv, ml) in um
        :return: Array[um:float, (x, y, 1), N]
        """
        cx = self.width / 2
        cy = self.height / 2
        q[0] = origin[0] - q[0] * 1000
        q[1] = origin[1] + q[1] * 1000
        q[2] = origin[2] + q[2] * 1000
        q = self.brain_view.project(q.T, um=True).T  # Array[int, (p,x,y), N]
        q = q[(1, 2, 0), :] * self.brain_view.resolution  # Array[float, (x,y,p), N]
        q[0] = q[0] - cx
        q[1] = cy - q[1]
        q[2] = 1
        return q

    # ================== #
    # SliceView updating #
    # ================== #

    def update_brain_view(self, view: SLICE | SliceView | str):
        if is_recursive_called():
            return

        old_state = self.get_boundary_state()

        if isinstance(view, str):
            try:
                view = SliceView(self.brain, view)
            except ValueError as e:
                self.logger.warning('slice_view(%s)', view, exc_info=e)
                view = SliceView(self.brain, 'coronal')

        self._brain_view = view
        self.logger.debug('slice_view(%s)', view.name)

        if (p := self._brain_slice) is not None:
            p = view.plane_at(p.coor_on())
        else:
            p = view.plane_at(view.n_plane // 2)

        self._brain_slice = None  # avoid self.plane_slider.value invoke updating methods
        self.update_brain_slice(p, update_image=False)
        self.update_boundary_transform(s=(old_state['sx'], old_state['sy']))

    # =================== #
    # SlicePlane updating #
    # =================== #

    def update_brain_slice(self, plane: int | SlicePlane | None, *,
                           update_image=True):
        if is_recursive_called():
            return

        view = self.brain_view

        if isinstance(plane, int):
            if (_slice := self._brain_slice) is not None:
                plane = _slice.with_plane(plane)
            else:
                plane = view.plane(plane)

        self._brain_slice: SlicePlane = plane
        # self.logger.debug('slice_plane(%d)', plane.plane)

        try:
            self.slice_select.value = view.name
            self.plane_slider.step = view.resolution

            if view.name == 'sagittal':
                d = view.n_plane * view.resolution / 2
                self.plane_slider.start = -d
                self.plane_slider.end = d
            else:
                self.plane_slider.start = 0
                self.plane_slider.end = view.n_plane * view.resolution

            self.rotate_hor_slider.step = view.resolution
            self.rotate_ver_slider.step = view.resolution
        except AttributeError:
            pass

        if plane is not None:
            try:
                if view.name == 'sagittal':
                    d = view.n_plane * view.resolution / 2
                    self.plane_slider.value = plane.plane * view.resolution - d
                else:
                    self.plane_slider.value = plane.plane * view.resolution

                self.rotate_hor_slider.value = plane.dw * view.resolution
                self.rotate_ver_slider.value = plane.dh * view.resolution
            except AttributeError:
                pass

        if update_image:
            if plane is None:
                self.update_image(None)
            else:
                self.update_image(plane.image)

    # ================= #
    # boundary updating #
    # ================= #

    def on_boundary_transform(self, state: BoundaryState):
        super().on_boundary_transform(state)

        if (plane := self._brain_slice) is not None:
            self.update_image(plane.image)

    # ============== #
    # image updating #
    # ============== #

    def update_image(self, image_data: NDArray[np.uint] | None):
        if image_data is None:
            self.visible = False
            self.data_brain.data = dict(image=[], dw=[], dh=[], x=[], y=[])
        else:
            self.data_brain.data = self.transform_image_data(np.flipud(image_data))

        self.update_region_image()
        self.update_label_position()

    def update_region_image(self):
        if len(self._regions) == 0 or (plane := self._brain_slice) is None:
            self.data_region.data = dict(image=[], dw=[], dh=[], x=[], y=[])
        else:
            view = SliceView(self.brain, plane.slice.name, self.brain.annotation)
            plane = view.plane_at(plane)
            self.data_region.data = self.transform_image_data(self.process_image_data(np.flipud(plane.image)))

    def process_image_data(self, image: NDArray[np.uint]) -> NDArray[np.uint]:
        return self._structure.image_annotation(image, self._regions, 0)
