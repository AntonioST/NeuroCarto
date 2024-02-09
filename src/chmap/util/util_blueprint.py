import numpy as np
from numpy.typing import NDArray

__all__ = ['BlueprintFunctions']


# noinspection PyMethodMayBeStatic
class BlueprintFunctions:
    """
    Provide blueprint manipulating functions.
    It is used by `chmap.views.edit_blueprint.CriteriaParser`.
    """

    def __init__(self, s: NDArray[np.int_], x: NDArray[np.int_], y: NDArray[np.int_]):
        self._s = s
        self._x = x
        self._y = y

    def id(self, v):
        return v

    def move(self, a: NDArray, *, x: int = 0, y: int = 0, s: int = 0):
        pass
