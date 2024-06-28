from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas

__all__ = ['get_atlas_brain', 'BrainGlobeAtlas', 'REFERENCE']


def get_atlas_brain(source: int | str = 25, cache_dir: str | Path = None, *,
                    check_latest=False) -> BrainGlobeAtlas:
    from brainglobe_atlasapi import BrainGlobeAtlas

    if isinstance(source, int):
        source = f"allen_mouse_{source}um"

    if cache_dir is not None:
        cache_dir = str(Path(cache_dir).absolute())

    return BrainGlobeAtlas(
        source,
        brainglobe_dir=cache_dir,
        # interm_download_dir=str(BRAIN_DIR.absolute()),
        check_latest=check_latest,
    )


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
