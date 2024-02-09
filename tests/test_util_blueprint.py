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

    def build_function(self, m: NDArray[np.int_],
                       categories: dict[str, int] = None,
                       xy: tuple[int, int, int, int] = (1, 0, 1, 0)) -> BlueprintFunctions:
        s, y, x = m.shape
        n = x * y
        y, x = np.mgrid[0:y, 0:x]
        x = np.tile(x.ravel(), s) * xy[0] + xy[1]
        y = np.tile(y.ravel(), s) * xy[2] + xy[3]
        s = np.repeat(np.arange(s), n)

        c = dict(self.DEFAULT_CATEGORIES)
        if categories is not None:
            c.update(categories)
        return BlueprintFunctions(s, x, y, c)

    def test_build(self):
        bp = self.build_function(np.array([
            0, 0,
            0, 0,
            0, 0,
            0, 0,
        ]).reshape(1, 4, 2))

        assert_array_equal(bp.s, np.array([0, 0, 0, 0, 0, 0, 0, 0]))
        assert_array_equal(bp.x, np.array([0, 1, 0, 1, 0, 1, 0, 1]))
        assert_array_equal(bp.y, np.array([0, 0, 1, 1, 2, 2, 3, 3]))
        self.assertEqual(bp.dx, 1)
        self.assertEqual(bp.dy, 1)

        bp = self.build_function(np.array([
            0, 0,
            0, 0,
            0, 0,
            0, 0,
        ]).reshape(2, 2, 2))

        assert_array_equal(bp.s, np.array([0, 0, 0, 0, 1, 1, 1, 1]))
        assert_array_equal(bp.x, np.array([0, 1, 0, 1, 0, 1, 0, 1]))
        assert_array_equal(bp.y, np.array([0, 0, 1, 1, 0, 0, 1, 1]))
        self.assertEqual(bp.dx, 1)
        self.assertEqual(bp.dy, 1)

    def test_move(self):
        blueprint = np.array([
            2, 0,
            1, 2,
            0, 1,
            # ----
            4, 0,
            3, 4,
            0, 3,
        ])

        bp = self.build_function(blueprint.reshape(2, 3, 2))
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
        blueprint = np.array([
            2, 0,
            np.nan, 3,
            2, 0,
            # ----
            2, 3,
            3, np.nan,
            2, 1,
        ], dtype=float)

        bp = self.build_function(blueprint.reshape(2, 3, 2))
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
        blueprint = np.array([
            2, 2, 2,
            1, 2, 3,
            1, 1, 3,
        ])

        bp = self.build_function(blueprint.reshape(1, 3, 3))
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

    def test_fill(self):
        blueprint = np.array([
            1, 1,
            0, 1,
            1, 1,
            0, 1,
            1, 1,
        ])

        bp = self.build_function(blueprint.reshape(1, 2, 5))
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


if __name__ == '__main__':
    unittest.main()
