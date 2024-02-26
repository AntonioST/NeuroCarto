from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._atlas_brain_type import BrainGlobeAtlas
else:
    BrainGlobeAtlas = Any

__all__ = ['get_atlas_brain', 'BrainGlobeAtlas']


def get_atlas_brain(source: int | str = 25, cache_dir: str | Path = None, *,
                    check_latest=False) -> BrainGlobeAtlas:
    from bg_atlasapi import BrainGlobeAtlas

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



