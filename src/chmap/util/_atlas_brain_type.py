from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict, overload, TypeAlias

import numpy as np
from numpy.typing import NDArray

__all__ = ['BrainGlobeAtlas']

JSON: TypeAlias = dict[str, Any]
COORD: TypeAlias = tuple[float, float, float] | list[float] | NDArray[np.float_]
MESH: TypeAlias = Any
ACRONYM: TypeAlias = str
MESH: TypeAlias = Any  #


class Structure(TypedDict):
    """Type annotation for :class:`bg_atlasapi.structure_class.Structure`"""
    acronym: str
    id: int
    name: str
    structure_id_path: list[int]
    rgb_triplet: list[int]
    mesh_filename: Path
    mesh: MESH


class Atlas:
    """Base class to handle atlases in BrainGlobe.

    Type annotation for :class:`bg_atlasapi.core.Atlas`
    """

    left_hemisphere_value = 1
    right_hemisphere_value = 2

    root_dir: Path
    metadata: JSON
    structures_list: JSON

    structures: dict[int | str, Structure]
    space: Any  # AnatomicalSpace

    def __init__(self, path: str | Path):
        """

        :param path: Path to folder containing data info.
        """

    @property
    def resolution(self) -> tuple[int, int, int]: ...

    @property
    def orientation(self) -> float: ...

    @property
    def shape(self) -> tuple[int, int, int]: ...

    @property
    def shape_um(self) -> tuple[int, int, int]: ...

    @property
    def hierarchy(self) -> 'treelib.tree.Tree':
        """Returns a :class:`treelib.tree.Tree` object with structures hierarchy."""

    @property
    def lookup_df(self):
        """Returns a dataframe with id, acronym and name for each structure."""

    @property
    def reference(self) -> NDArray[np.uint]: ...

    @property
    def annotation(self) -> NDArray[np.uint]: ...

    @property
    def hemispheres(self) -> NDArray[np.uint]: ...

    def hemisphere_from_coords(self, coords: COORD, microns: bool = False, as_string: bool = False) -> int | ACRONYM:
        """Get the hemisphere from a coordinate triplet.

        :param coords: Triplet of coordinates. Default in voxels, can be microns if microns=True
        :param microns: If true, coordinates are interpreted in microns.
        :param as_string: If true, returns "left" or "right".
        :return: Hemisphere label.
        """

    def structure_from_coords(self, coords: COORD, microns: bool = False, as_acronym: bool = False,
                              hierarchy_lev=None) -> int | ACRONYM:
        """Get the structure from a coordinate triplet.

        :param coords: Triplet of coordinates.
        :param microns: If true, coordinates are interpreted in microns.
        :param as_acronym: If true, the region acronym is returned.
        :param hierarchy_lev: If specified, return parent node at thi hierarchy level.
        :return: Structure containing the coordinates.
        """

    @overload
    def mesh_from_structure(self, structure: int | ACRONYM) -> MESH: ...

    @overload
    def mesh_from_structure(self, structure: list[int]) -> list[MESH]: ...

    def mesh_from_structure(self, structure): ...

    @overload
    def meshfile_from_structure(self, structure: int | ACRONYM) -> Path: ...

    @overload
    def meshfile_from_structure(self, structure: list[int]) -> list[Path]: ...

    def meshfile_from_structure(self, structure): ...

    def root_mesh(self) -> MESH: ...

    def root_meshfile(self) -> Path: ...

    def get_structure_ancestors(self, structure: int | ACRONYM) -> list[ACRONYM]:
        """Returns a list of acronyms for all ancestors of a given structure

        :param structure: Structure id or acronym
        :return: List of descendants acronyms
        """

    def get_structure_descendants(self, structure: int | ACRONYM) -> list[ACRONYM]:
        """Returns a list of acronyms for all descendants of a given structure.

        :param structure: Structure id or acronym
        :return: List of descendants acronyms
        """

    def get_structure_mask(self, structure: int | ACRONYM) -> NDArray[np.bool_]:
        """Returns a stack with the mask for a specific structure (including all
        sub-structures). This function is not particularly optimized, and might
        take some hundreds of ms per run for some structures.

        :param structure: Structure id or acronym
        :return: stack containing the mask array.
        """


# noinspection PyPropertyDefinition,PyUnusedLocal
class BrainGlobeAtlas(Atlas):
    """Add remote atlas fetching and version comparison functionalities
    to the core Atlas class.

    Type annotation for :class:`bg_atlasapi.BrainGlobeAtlas`
    """

    atlas_name: str

    def __init__(self,
                 atlas_name: str,
                 brainglobe_dir: Path = None,
                 interm_download_dir: Path = None,
                 check_latest: bool = True,
                 print_authors: bool = True):
        """

        :param atlas_name: Name of the atlas to be used.
        :param brainglobe_dir: Default folder for brainglobe downloads.
        :param interm_download_dir: Folder to download the compressed file for extraction.
        :param check_latest: If true, check if we have the most recent atlas (default=True). Set
                this to False to avoid waiting for remote server response on atlas
                instantiation and to suppress warnings.
        :param print_authors: If true, disable default listing of the atlas reference.
        """

    @property
    def local_version(self) -> tuple[int, ...] | None:
        """If atlas is local, return actual version of the downloaded files;
        Else, return None.
        """

    @property
    def remote_version(self) -> tuple[int, ...] | None:
        """Remote version read from GIN conf file. If we are offline, return None.
        """

    @property
    def local_full_name(self) -> str | None:
        """As we can't know the local version a priori, search candidate dirs
        using name and not version number. If none is found, return None.
        """

    @property
    def remote_url(self) -> str | None:
        """Format complete url for download."""

    def download_extract_file(self) -> None:
        """Download and extract atlas from remote url."""

    def check_latest_version(self) -> None:
        """Checks if the local version is the latest available
        and prompts the user to update if not.
        """
