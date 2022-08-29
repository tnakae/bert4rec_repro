from typing import Generator, Iterable, TypeVar

T = TypeVar("T")


def generator_limit(generator: Iterable[T], n: int) -> Generator[T, None, None]:
    limit = 0
    for item in generator:
        if limit >= n:
            break
        yield item
        limit += 1
