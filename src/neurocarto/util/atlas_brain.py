from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas

__all__ = [
    'brainglobe_atlasapi_version',
    'get_atlas_brain',
    'get_atlas_brain_reference',
    'BrainGlobeAtlas',
    'REFERENCE'
]

_BRAINGLOBE_ATLASAPI_VERSION = None


def brainglobe_atlasapi_version() -> int:
    global _BRAINGLOBE_ATLASAPI_VERSION
    if _BRAINGLOBE_ATLASAPI_VERSION is None:
        import logging
        logger = logging.getLogger('neurocarto.atlas')

        import brainglobe_atlasapi
        version = brainglobe_atlasapi.__version__
        logger.debug('brainglobe_atlasapi.version=%s', version)

        m, *_ = version.split('.')
        _BRAINGLOBE_ATLASAPI_VERSION = int(m)

    return _BRAINGLOBE_ATLASAPI_VERSION


def get_atlas_name_list() -> tuple[str, ...]:
    from typing import get_args
    from brainglobe_atlasapi import atlas_name
    return get_args(atlas_name.AtlasName)


def get_atlas_brain(source: int | str = 25, cache_dir: str | Path = None, *,
                    check_latest=False) -> BrainGlobeAtlas:
    from brainglobe_atlasapi import BrainGlobeAtlas

    if isinstance(source, int):
        source = f"allen_mouse_{source}um"

    if cache_dir is not None:
        cache_dir = str(Path(cache_dir).absolute())

    if source not in get_atlas_name_list():
        raise RuntimeError(f'Not a valid atlas name: {source}')

    return BrainGlobeAtlas(
        source,
        brainglobe_dir=cache_dir,
        # interm_download_dir=str(BRAIN_DIR.absolute()),
        check_latest=check_latest,
    )


def get_atlas_brain_reference(atlas: BrainGlobeAtlas) -> NDArray[np.uint]:
    if brainglobe_atlasapi_version() >= 3:
        return atlas.template
    else:
        return atlas.reference


REFERENCE = {
    # name: {}
    'bregma': {
        # name: (ap, dv, ml) um
        # (540 0 570) index for 10 um
        'allen_mouse_10um': (5400, 0, 5700),
        'allen_mouse_25um': (5400, 0, 5700),
        'allen_mouse_50um': (5400, 0, 5700),
    }
}
