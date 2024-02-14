from typing import Any, TypeVar, overload

__all__ = ['get_value']

T = TypeVar('T')


@overload
def get_value(desp: str, args: list[Any], i: int) -> str:
    pass


@overload
def get_value(desp: str, args: list[Any], i: int, t: type[T] = str) -> T:
    pass


def get_value(desp: str, args: list[Any], i: int, t: type[T] = str) -> T:
    try:
        v = args[i]
    except IndexError as e:
        raise ValueError(f'expect {desp} at {i}-th argument') from e

    try:
        return t(v)
    except ValueError as e:
        raise ValueError(f'expect {desp} is type {t.__name__} at {i}-th argument, but {type(v).__name__}') from e
