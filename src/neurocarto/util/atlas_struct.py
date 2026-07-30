from __future__ import annotations

from typing import NamedTuple, Iterator, Self, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from brainglobe_atlasapi import BrainGlobeAtlas, StructuresDict, Structure as _Structure

__all__ = ['Structures', 'Structure']

ACRONYM = str
STRUCTURE = int


class Structure(NamedTuple):
    acronym: ACRONYM
    id: STRUCTURE
    name: str
    color: tuple[int, int, int]
    id_path: tuple[STRUCTURE, ...]

    @classmethod
    def of(cls, struct: _Structure) -> Self:
        return Structure(struct['acronym'], struct['id'], struct['name'], tuple(struct['rgb_triplet']), tuple(struct['structure_id_path']))


class Structures:
    def __init__(self, brain: BrainGlobeAtlas | StructuresDict):
        from brainglobe_atlasapi import BrainGlobeAtlas
        if isinstance(brain, BrainGlobeAtlas):
            brain = brain.structures

        self._structure: dict[ACRONYM, Structure] = {
            ac: Structure.of(brain[it])
            for ac, it in brain.acronym_to_id_map.items()
        }

    def __getitem__(self, item: STRUCTURE | ACRONYM) -> Structure:
        if isinstance(item, str):
            return self._structure[item]

        elif isinstance(item, int):
            for _, structure in self._structure.items():
                if structure.id == item:
                    return structure

            raise KeyError(item)
        else:
            raise TypeError()

    def __iter__(self) -> Iterator[Structure]:
        return iter(self._structure.values())

    @property
    def regions(self) -> list[ACRONYM]:
        return list(self._structure)

    def alias(self, acronym: ACRONYM, *name: str):
        for it in name:
            self._structure[it] = self._structure[acronym]

    def is_subregion(self, parent: STRUCTURE | ACRONYM | Structure, child: STRUCTURE | ACRONYM | Structure) -> bool:
        """
        Do both have relationship: parent / child?

        :param parent:
        :param child:
        :return:
        """
        if isinstance(parent, (int, str)):
            parent = self[parent]
        if isinstance(child, (int, str)):
            child = self[child]
        return parent.id in child.id_path

    def iter_subregions(self, region: STRUCTURE | ACRONYM | Structure) -> Iterator[Structure]:
        if isinstance(region, (int, str)):
            region = self[region]

        for it in self._structure.values():
            if region.id in it.id_path:
                yield it

    def sort_structure(self, regions: list[STRUCTURE | ACRONYM]) -> list[STRUCTURE | ACRONYM]:
        """
        For sorted list, two elements r[i] and r[j] for i, j in N, i < j,
        that either r[i]/r[j] or not(r[i]/r[j]).


        :param regions:
        :return:
        """
        return list(sorted(regions, key=lambda it: len(self[it].id_path)))

    def image_annotation(self, annotation: NDArray[np.uint],
                         merge: dict[STRUCTURE | ACRONYM, int],
                         other: int) -> NDArray[np.uint]:
        ret = np.full_like(annotation, other)
        for region in self.sort_structure(list(merge)):
            for sub in self.iter_subregions(region):
                ret[annotation == sub.id] = merge[region]
        return ret
