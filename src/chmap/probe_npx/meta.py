from typing import TypedDict

__all__ = ['NpxMeta']


class NpxMeta(TypedDict, total=False):
    serial_number: str  # imDatPrb_sn
    imro_table: str  # ~imroTbl
