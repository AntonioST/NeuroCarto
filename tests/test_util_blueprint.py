import textwrap
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

from neurocarto.probe_npx.desp import NpxProbeDesp
from neurocarto.util.util_blueprint import BlueprintFunctions

DEFAULT_CATEGORIES = {
    'UNSET': 0,
    'FORBIDDEN': 1,
    'LOW': 2,
    'SET': 3,
}


# noinspection PyFinal
def bp_from_shape(shape: tuple[int, int, int]) -> BlueprintFunctions:
    """

    :param shape: (shank, row, col)
    :return:
    """
    s, y, x = shape
    n = x * y
    yy, xx = np.mgrid[0:y, 0:x]
    xx = np.tile(xx.ravel(), s)
    yy = np.tile(yy.ravel(), s)
    ss = np.repeat(np.arange(s), n)

    ret = object.__new__(BlueprintFunctions)
    ret.probe = NpxProbeDesp()
    ret.channelmap = None
    ret.categories = DEFAULT_CATEGORIES
    ret.s = ss
    ret.x = xx
    ret.y = yy
    ret.dx = 1
    ret.dy = 1
    ret._position_index = {
        (int(ss[i]), int(xx[i]), int(yy[i])): i
        for i in range(len(ss))
    }
    ret._blueprint = np.zeros_like(s)
    return ret


class UtilBlueprintTest(unittest.TestCase):

    def test_build(self):
        bp = bp_from_shape((1, 4, 2))

        assert_array_equal(bp.s, np.array([0, 0, 0, 0, 0, 0, 0, 0]))
        assert_array_equal(bp.x, np.array([0, 1, 0, 1, 0, 1, 0, 1]))
        assert_array_equal(bp.y, np.array([0, 0, 1, 1, 2, 2, 3, 3]))
        self.assertEqual(bp.dx, 1)
        self.assertEqual(bp.dy, 1)

        bp = bp_from_shape((2, 2, 2))

        assert_array_equal(bp.s, np.array([0, 0, 0, 0, 1, 1, 1, 1]))
        assert_array_equal(bp.x, np.array([0, 1, 0, 1, 0, 1, 0, 1]))
        assert_array_equal(bp.y, np.array([0, 0, 1, 1, 0, 0, 1, 1]))
        self.assertEqual(bp.dx, 1)
        self.assertEqual(bp.dy, 1)

    def test_move(self):
        bp = bp_from_shape((2, 3, 2))

        blueprint = np.array([
            2, 0,
            1, 2,
            0, 1,
            # ----
            4, 0,
            3, 4,
            0, 3,
        ])

        assert_array_equal(bp.move(blueprint, ty=-1), np.array([
            1, 2,
            0, 1,
            0, 0,
            # ----
            3, 4,
            0, 3,
            0, 0,
        ]))
        assert_array_equal(bp.move(blueprint, tx=1), np.array([
            0, 2,
            0, 1,
            0, 0,
            # ----
            0, 4,
            0, 3,
            0, 0,
        ]))

    def test_interpolate_nan(self):
        bp = bp_from_shape((2, 3, 2))

        blueprint = np.array([
            2, 0,
            np.nan, 3,
            2, 0,
            # ----
            2, 3,
            3, np.nan,
            2, 1,
        ], dtype=float)

        assert_array_equal(bp.interpolate_nan(blueprint), np.array([
            2, 0,
            2, 3,
            2, 0,
            # ----
            2, 3,
            3, 2,
            2, 1,
        ], dtype=float))
        assert_array_equal(bp.interpolate_nan(blueprint, (1, 0)), np.array([
            2, 0,
            3, 3,
            2, 0,
            # ----
            2, 3,
            3, 3,
            2, 1,
        ], dtype=float))

    def assert_clustering(self, x: NDArray[np.int_], y: NDArray[np.int_]):
        for c in np.unique(x):
            if len(np.unique(y[x == c])) != 1:
                self.fail(textwrap.dedent(f"""\
                assert_clustering for cluster {c} in x
                x : {x}
                y : {y}
                """))

    def test_find_clustering(self):
        bp = bp_from_shape((1, 3, 3))

        blueprint = np.array([
            2, 2, 2,
            1, 2, 3,
            1, 1, 3,
        ])

        self.assert_clustering(bp.find_clustering(blueprint), np.array([
            2, 2, 2,
            1, 2, 3,
            1, 1, 3,
        ]))

        blueprint = np.array([
            2, 2, 1,
            1, 2, 1,
            1, 3, 1,
        ])

        self.assert_clustering(bp.find_clustering(blueprint), np.array([
            2, 2, 4,
            1, 2, 4,
            1, 3, 4,
        ]))

    def test_find_clustering_no_diagonal(self):
        bp = bp_from_shape((1, 3, 3))

        blueprint = np.array([
            2, 3, 3,
            1, 2, 3,
            1, 1, 2,
        ])

        self.assert_clustering(bp.find_clustering(blueprint, diagonal=False), np.array([
            2, 5, 5,
            1, 3, 5,
            1, 1, 4,
        ]))

    def test_clustering_edges_single(self):
        bp = bp_from_shape((1, 3, 3))
        blueprint = np.array([
            0, 0, 0,
            0, 1, 0,
            0, 0, 0,
        ])
        results = bp.clustering_edges(blueprint, categories=[1])
        self.assertEqual(1, len(results))
        result = results[0]
        self.assertEqual(1, result.category)
        self.assertEqual(0, result.shank)
        self.assertListEqual([(1, 1, 1), (1, 1, 3), (1, 1, 5), (1, 1, 7)], result.edges)

    def test_clustering(self):
        bp = bp_from_shape((1, 4, 4))
        blueprint = np.array([
            0, 0, 0, 0,
            0, 1, 1, 0,
            0, 1, 1, 0,
            0, 0, 0, 0,
        ])
        results = bp.clustering_edges(blueprint, categories=[1])
        self.assertEqual(1, len(results))
        result = results[0]
        self.assertEqual(1, result.category)
        self.assertEqual(0, result.shank)
        # 5 6 7
        # 4 e 0
        # 3 2 1
        self.assertListEqual([(1, 1, 5), (1, 1, 6), (2, 1, 7), (2, 2, 1), (1, 2, 3)], result.edges)
        self.assertListEqual([(1, 1, 8), (2, 1, 8), (2, 2, 8), (1, 2, 8)], result.set_corner((0, 0)).edges)

    def test_clustering_hole(self):
        bp = bp_from_shape((1, 5, 5))
        blueprint = np.array([
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            0, 1, 0, 1, 0,
            0, 1, 1, 1, 0,
            0, 0, 0, 0, 0,
        ])
        results = bp.clustering_edges(blueprint, categories=[1])
        self.assertEqual(1, len(results))
        result = results[0]
        self.assertEqual(1, result.category)
        self.assertEqual(0, result.shank)
        # 5 6 7
        # 4 e 0
        # 3 2 1
        self.assertListEqual([(1, 1, 8), (3, 1, 8), (3, 3, 8), (1, 3, 8)], result.set_corner((0, 0)).edges)

    def test_edge_rastering(self):
        bp = bp_from_shape((1, 5, 5))
        blueprint = np.array([
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            0, 1, 0, 1, 0,
            0, 1, 1, 1, 0,
            0, 0, 0, 0, 0,
        ])
        results = bp.clustering_edges(blueprint, categories=[1])
        assert_array_equal(blueprint, bp.edge_rastering(results, fill=False))

        blueprint = np.array([
            0, 1, 1, 0, 0,
            1, 1, 1, 1, 0,
            1, 0, 0, 1, 0,
            1, 1, 1, 1, 0,
            0, 0, 0, 1, 1,
        ])
        results = bp.clustering_edges(blueprint, categories=[1])

        assert_array_equal(blueprint, bp.edge_rastering(results, fill=False))

    def test_edge_rastering_fill(self):
        bp = bp_from_shape((1, 5, 5))
        blueprint = np.array([
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            0, 1, 0, 1, 0,
            0, 1, 1, 1, 0,
            0, 0, 0, 0, 0,
        ])
        results = bp.clustering_edges(blueprint, categories=[1])

        blueprint_expected = np.array([
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            0, 0, 0, 0, 0,
        ])
        assert_array_equal(blueprint_expected, bp.edge_rastering(results, fill=True))

        blueprint = np.array([
            0, 1, 1, 0, 0,
            1, 1, 1, 1, 0,
            1, 0, 0, 1, 0,
            1, 1, 1, 1, 0,
            0, 0, 0, 1, 1,
        ])
        results = bp.clustering_edges(blueprint, categories=[1])

        blueprint_expected = np.array([
            0, 1, 1, 0, 0,
            1, 1, 1, 1, 0,
            1, 1, 1, 1, 0,
            1, 1, 1, 1, 0,
            0, 0, 0, 1, 1,
        ])
        assert_array_equal(blueprint_expected, bp.edge_rastering(results, fill=True))

    def test_edge_rastering_cover(self):
        bp = bp_from_shape((1, 5, 5))
        b1 = np.array([
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 0,
            0, 0, 0, 0, 0,
        ])
        b2 = np.array([
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0,
            0, 0, 2, 2, 2,
            0, 0, 2, 2, 2,
            0, 0, 2, 2, 2,
        ])

        r1 = bp.clustering_edges(b1, [1])[0]
        r2 = bp.clustering_edges(b2, [2])[0]

        blueprint_expected = np.array([
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            0, 1, 2, 2, 2,
            0, 1, 2, 2, 2,
            0, 0, 2, 2, 2,
        ])
        assert_array_equal(blueprint_expected, bp.edge_rastering([r1, r2], fill=True, overwrite=True))

        blueprint_expected = np.array([
            0, 0, 0, 0, 0,
            0, 1, 1, 1, 0,
            0, 1, 1, 1, 2,
            0, 1, 1, 1, 2,
            0, 0, 2, 2, 2,
        ])
        assert_array_equal(blueprint_expected, bp.edge_rastering([r1, r2], fill=True, overwrite=False))

    def test_fill(self):
        bp = bp_from_shape((1, 5, 2))

        blueprint = np.array([
            1, 1,
            0, 1,
            1, 1,
            0, 1,
            1, 1,
        ])

        self.assert_clustering(bp.fill(blueprint), np.array([
            1, 1,
            1, 1,
            1, 1,
            1, 1,
            1, 1,
        ]))

        blueprint = np.array([
            0, 1,
            1, 1,
            1, 1,
            1, 1,
            1, 0,
        ])
        self.assert_clustering(bp.fill(blueprint), np.array([
            1, 1,
            1, 1,
            1, 1,
            1, 1,
            1, 1,
        ]))

        blueprint = np.array([
            0, 0,
            0, 1,
            1, 1,
            1, 0,
            0, 0,
        ])
        self.assert_clustering(bp.fill(blueprint), np.array([
            0, 0,
            1, 1,
            1, 1,
            1, 1,
            0, 0,
        ]))

    def test_fill_with_threshold(self):
        bp = bp_from_shape((1, 5, 2))

        blueprint = np.array([
            2, 2,
            2, 2,
            2, 0,
            0, 1,
            1, 1,
        ])

        self.assert_clustering(bp.fill(blueprint, threshold=4), np.array([
            2, 2,
            2, 2,
            2, 2,
            0, 1,
            1, 1,
        ]))

        blueprint = np.array([
            2, 2,
            2, 2,
            2, 0,
            0, 1,
            1, 1,
        ])
        self.assert_clustering(bp.fill(blueprint, threshold=4, unset=True), np.array([
            2, 2,
            2, 2,
            2, 2,
            0, 0,
            0, 0,
        ]))

    def test_fill_rect(self):
        bp = bp_from_shape((1, 7, 2))

        blueprint = np.array([
            0, 0,
            0, 1,
            0, 1,
            1, 1,
            1, 0,
            1, 0,
            0, 0,
        ])

        self.assert_clustering(bp.fill(blueprint, gap=None), np.array([
            0, 0,
            1, 1,
            1, 1,
            1, 1,
            1, 1,
            1, 1,
            0, 0,
        ]))

    def test_extend(self):
        bp = bp_from_shape((1, 5, 2))

        blueprint = np.array([
            0, 0,
            0, 0,
            1, 1,
            0, 0,
            0, 0,
        ])

        self.assert_clustering(bp.extend(blueprint, category=1, step=1), np.array([
            0, 0,
            1, 1,
            1, 1,
            1, 1,
            0, 0,
        ]))

        blueprint = np.array([
            0, 0,
            0, 0,
            1, 1,
            0, 0,
            0, 0,
        ])
        self.assert_clustering(bp.extend(blueprint, category=1, step=1, value=2), np.array([
            0, 0,
            2, 2,
            1, 1,
            2, 2,
            0, 0,
        ]))

    def test_extend_direction(self):
        bp = bp_from_shape((1, 5, 2))

        blueprint = np.array([
            0, 0,
            0, 0,
            1, 1,
            0, 0,
            0, 0,
        ])

        self.assert_clustering(bp.extend(blueprint, category=1, step=1, bi=False), np.array([
            0, 0,
            0, 0,
            1, 1,
            1, 1,
            0, 0,
        ]))

        blueprint = np.array([
            0, 0,
            0, 0,
            1, 1,
            0, 0,
            0, 0,
        ])
        self.assert_clustering(bp.extend(blueprint, category=1, step=-1, bi=False), np.array([
            0, 0,
            1, 1,
            1, 1,
            0, 0,
            0, 0,
        ]))

        blueprint = np.array([
            0, 0,
            0, 0,
            1, 0,
            0, 0,
            0, 0,
        ])
        self.assert_clustering(bp.extend(blueprint, category=1, step=1), np.array([
            0, 0,
            1, 0,
            1, 0,
            1, 0,
            0, 0,
        ]))
        blueprint = np.array([
            0, 0,
            0, 0,
            1, 0,
            0, 0,
            0, 0,
        ])
        self.assert_clustering(bp.extend(blueprint, category=1, step=(1, 1)), np.array([
            0, 0,
            1, 1,
            1, 1,
            1, 1,
            0, 0,
        ]))

    def test_extend_threshold(self):
        bp = bp_from_shape((1, 5, 2))

        blueprint = np.array([
            1, 1,
            0, 0,
            0, 0,
            1, 1,
            1, 1,
        ])
        self.assert_clustering(bp.extend(blueprint, category=1, step=1, threshold=4), np.array([
            1, 1,
            0, 0,
            1, 1,
            1, 1,
            1, 1,
        ]))

    def test_extend_overwrite(self):
        bp = bp_from_shape((1, 5, 2))

        blueprint = np.array([
            2, 2,
            0, 1,
            1, 1,
            0, 1,
            0, 0,
        ])
        self.assert_clustering(bp.extend(blueprint, category=1, step=1), np.array([
            2, 2,
            1, 1,
            1, 1,
            1, 1,
            0, 1,
        ]))

        blueprint = np.array([
            2, 2,
            0, 1,
            1, 1,
            0, 1,
            0, 0,
        ])
        self.assert_clustering(bp.extend(blueprint, category=1, step=1, overwrite=True), np.array([
            2, 1,
            1, 1,
            1, 1,
            1, 1,
            0, 1,
        ]))

    def test_reduce(self):
        bp = bp_from_shape((1, 7, 2))

        blueprint = np.array([
            0, 0,
            1, 1,
            1, 1,
            1, 1,
            1, 1,
            1, 1,
            0, 0,
        ])
        self.assert_clustering(bp.reduce(blueprint, category=1, step=1), np.array([
            0, 0,
            0, 0,
            1, 1,
            1, 1,
            1, 1,
            0, 0,
            0, 0,
        ]))

    def test_reduce_small_area(self):
        bp = bp_from_shape((1, 4, 2))

        blueprint = np.array([
            0, 0,
            1, 1,
            1, 1,
            0, 0,
        ])
        self.assert_clustering(bp.reduce(blueprint, category=1, step=1), np.array([
            0, 0,
            0, 0,
            0, 0,
            0, 0,
        ]))

    def test_reduce_direction(self):
        bp = bp_from_shape((1, 7, 2))

        blueprint = np.array([
            0, 0,
            1, 1,
            1, 1,
            1, 1,
            1, 1,
            1, 1,
            0, 0,
        ])
        # Because the origin is located at button, the reduce direction is reversed here.
        self.assert_clustering(bp.reduce(blueprint, category=1, step=1, bi=False), np.array([
            0, 0,
            0, 0,
            1, 1,
            1, 1,
            1, 1,
            1, 1,
            0, 0,
        ]))

        blueprint = np.array([
            0, 0,
            1, 1,
            1, 1,
            1, 1,
            1, 1,
            1, 1,
            0, 0,
        ])
        self.assert_clustering(bp.reduce(blueprint, category=1, step=-1, bi=False), np.array([
            0, 0,
            1, 1,
            1, 1,
            1, 1,
            1, 1,
            0, 0,
            0, 0,
        ]))

if __name__ == '__main__':
    unittest.main()
