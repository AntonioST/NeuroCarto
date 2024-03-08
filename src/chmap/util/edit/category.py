import numpy as np
from numpy.typing import NDArray

from chmap.util.util_blueprint import BlueprintFunctions

__all__ = ['mask', 'invalid']


def mask(self: BlueprintFunctions,
         blueprint: NDArray[np.int_],
         categories: int | list[int] = None) -> NDArray[np.bool_]:
    if categories is None:
        categories = list(self.categories.values())
        try:
            categories.remove(self.CATE_UNSET)
        except ValueError:
            pass
        try:
            categories.remove(self.CATE_FORBIDDEN)
        except ValueError:
            pass

    if isinstance(categories, (int, np.integer)):
        ret = blueprint == categories
    else:
        ret = np.zeros_like(blueprint, dtype=bool)
        for category in categories:
            np.logical_or(ret, blueprint == category, out=ret)

    return ret


def invalid(self: BlueprintFunctions,
            blueprint: NDArray[np.int_],
            electrodes: NDArray[np.int_] | NDArray[np.bool_],
            value: int = None, *,
            overwrite: bool = False) -> NDArray:
    if electrodes.dtype == np.bool_:
        blueprint[electrodes]  # check shape
        protected = electrodes
    else:
        protected = np.zeros_like(blueprint, dtype=bool)
        protected[electrodes] = True

    all_electrodes = self.electrodes
    electrodes = [all_electrodes[it] for it in np.nonzero(protected)[0]]

    invalid_electrodes = self.index_blueprint(self.probe.invalid_electrodes(self.channelmap, electrodes, all_electrodes))
    invalid_mask = np.zeros_like(protected, dtype=bool)
    invalid_mask[invalid_electrodes] = True

    if value is None:
        if not overwrite:
            invalid_mask[protected] = False
        return invalid_mask
    else:
        ret = blueprint.copy()

        if overwrite:
            ret[invalid_mask] = value
        else:
            ret[invalid_mask & ~protected] = value

        return ret
