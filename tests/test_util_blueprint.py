import textwrap
import unittest

import numpy as np
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

from chmap.util.util_blueprint import BlueprintFunctions


class UtilBlueprintTest(unittest.TestCase):
    DEFAULT_CATEGORIES = {
        'UNSET': 0,
        'FORBIDDEN': 1,
        'LOW': 2,
        'SET': 3,
    }

    def test_build(self):
        bp = BlueprintFunctions.from_shape((1, 4, 2), self.DEFAULT_CATEGORIES)

        assert_array_equal(bp.s, np.array([0, 0, 0, 0, 0, 0, 0, 0]))
        assert_array_equal(bp.x, np.array([0, 1, 0, 1, 0, 1, 0, 1]))
        assert_array_equal(bp.y, np.array([0, 0, 1, 1, 2, 2, 3, 3]))
        self.assertEqual(bp.dx, 1)
        self.assertEqual(bp.dy, 1)

        bp = BlueprintFunctions.from_shape((2, 2, 2), self.DEFAULT_CATEGORIES)

        assert_array_equal(bp.s, np.array([0, 0, 0, 0, 1, 1, 1, 1]))
        assert_array_equal(bp.x, np.array([0, 1, 0, 1, 0, 1, 0, 1]))
        assert_array_equal(bp.y, np.array([0, 0, 1, 1, 0, 0, 1, 1]))
        self.assertEqual(bp.dx, 1)
        self.assertEqual(bp.dy, 1)

    def test_move(self):
        bp = BlueprintFunctions.from_shape((2, 3, 2), self.DEFAULT_CATEGORIES)

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
        bp = BlueprintFunctions.from_shape((2, 3, 2), self.DEFAULT_CATEGORIES)

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
        bp = BlueprintFunctions.from_shape((1, 3, 3), self.DEFAULT_CATEGORIES)

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
        bp = BlueprintFunctions.from_shape((1, 3, 3), self.DEFAULT_CATEGORIES)

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
        bp = BlueprintFunctions.from_shape((1, 3, 3), self.DEFAULT_CATEGORIES)
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
        bp = BlueprintFunctions.from_shape((1, 4, 4), self.DEFAULT_CATEGORIES)
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
        bp = BlueprintFunctions.from_shape((1, 5, 5), self.DEFAULT_CATEGORIES)
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

    def test_fill(self):
        bp = BlueprintFunctions.from_shape((1, 5, 2), self.DEFAULT_CATEGORIES)

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
        bp = BlueprintFunctions.from_shape((1, 5, 2), self.DEFAULT_CATEGORIES)

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
        bp = BlueprintFunctions.from_shape((1, 7, 2), self.DEFAULT_CATEGORIES)

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
        bp = BlueprintFunctions.from_shape((1, 5, 2), self.DEFAULT_CATEGORIES)

        blueprint = np.array([
            0, 0,
            0, 0,
            1, 1,
            0, 0,
            0, 0,
        ])

        self.assert_clustering(bp.extend(blueprint, on=1, step=1), np.array([
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
        self.assert_clustering(bp.extend(blueprint, on=1, step=1, category=2), np.array([
            0, 0,
            2, 2,
            1, 1,
            2, 2,
            0, 0,
        ]))

    def test_extend_direction(self):
        bp = BlueprintFunctions.from_shape((1, 5, 2), self.DEFAULT_CATEGORIES)

        blueprint = np.array([
            0, 0,
            0, 0,
            1, 1,
            0, 0,
            0, 0,
        ])

        self.assert_clustering(bp.extend(blueprint, on=1, step=1, bi=False), np.array([
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
        self.assert_clustering(bp.extend(blueprint, on=1, step=-1, bi=False), np.array([
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
        self.assert_clustering(bp.extend(blueprint, on=1, step=1), np.array([
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
        self.assert_clustering(bp.extend(blueprint, on=1, step=(1, 1)), np.array([
            0, 0,
            1, 1,
            1, 1,
            1, 1,
            0, 0,
        ]))

    def test_extend_threshold(self):
        bp = BlueprintFunctions.from_shape((1, 5, 2), self.DEFAULT_CATEGORIES)

        blueprint = np.array([
            1, 1,
            0, 0,
            0, 0,
            1, 1,
            1, 1,
        ])
        self.assert_clustering(bp.extend(blueprint, on=1, step=1, threshold=4), np.array([
            1, 1,
            0, 0,
            1, 1,
            1, 1,
            1, 1,
        ]))

    def test_extend_overwrite(self):
        bp = BlueprintFunctions.from_shape((1, 5, 2), self.DEFAULT_CATEGORIES)

        blueprint = np.array([
            2, 2,
            0, 1,
            1, 1,
            0, 1,
            0, 0,
        ])
        self.assert_clustering(bp.extend(blueprint, on=1, step=1), np.array([
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
        self.assert_clustering(bp.extend(blueprint, on=1, step=1, overwrite=True), np.array([
            2, 1,
            1, 1,
            1, 1,
            1, 1,
            0, 1,
        ]))


if __name__ == '__main__':
    unittest.main()
