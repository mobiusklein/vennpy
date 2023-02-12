from dataclasses import dataclass
from itertools import combinations
from typing import ClassVar, List, Set, Iterable, Union, TypeVar, Protocol, runtime_checkable


T = TypeVar('T')

# Combinatoric Entities

@runtime_checkable
class SetLike(Protocol[T]):

    def get_elements(self) -> Set[T]:
        ...

    @property
    def elements(self):
        return self.get_elements()

    def overlap(self, other: Union[Set[T], List[T], 'SetLike[T]']) -> int:
        ...

    @property
    def cardinality(self) -> int:
        return len(self.get_elements())

    def __iter__(self):
        return iter(self.get_elements())


@dataclass
class BaseSet(SetLike[T]):
    name: str

    def overlap(self, other: Union[Set[T], List[T], 'SetLike[T]']):
        if isinstance(other, list):
            other = set(other)
        elif isinstance(other, SetLike):
            other = other.get_elements()

        return len(self.elements & other)

    @property
    def key(self):
        return f"{self.name}:{self.group_type}#{self.cardinality}"

    def union(self, *others: Iterable[Union[Set[T], List[T], SetLike[T]]]) -> 'SetUnion':
        return SetUnion(SetUnion.make_name([self, *others]), [self, *others])

    def intersection(self, *others: Iterable[Union[Set[T], List[T], SetLike[T]]]) -> 'SetIntersection':
        return SetIntersection(SetIntersection.make_name([self, *others]), [self, *others])

    def __and__(self, other: SetLike):
        return self.intersection(other)

    def __or__(self, other: SetLike):
        return self.union(other)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r}, {self.elements})"

    degree: ClassVar[int] = 1


@dataclass(repr=False)
class VSet(BaseSet[T]):

    _elements: Set[T]

    group_type: ClassVar[str] = 'set'

    def get_elements(self) -> Set[T]:
        return self._elements


@dataclass(repr=False)
class SetCombination(BaseSet[T]):
    sets: List[SetLike[T]]
    _symbol = "?"

    @classmethod
    def make_name(cls, sets: List[BaseSet]) -> str:
        return "(%s)" % f" {cls._symbol} ".join([s.name for s in sorted(sets, key=lambda x: x.cardinality)])

    @property
    def degree(self):
        return len(self.sets)

    def get_elements(self) -> Set[T]:
        raise NotImplementedError()


@dataclass(repr=False)
class SetIntersection(SetCombination[T]):
    group_type: ClassVar[str] = 'intersection'
    _symbol = '&'

    def get_elements(self) -> Set[T]:
        base = set(self.sets[0].get_elements())
        for s in self.sets[1:]:
            base &= s.get_elements()
        return base


@dataclass(repr=False)
class SetUnion(SetCombination[T]):
    group_type: ClassVar[str] = 'union'
    _symbol = '|'

    def get_elements(self) -> Set[T]:
        base = set(self.sets[0].get_elements())
        for s in self.sets[1:]:
            base |= s.get_elements()
        return base


def combinate_sets(sets: List[VSet[T]]) -> List[SetLike[T]]:
    max_order = len(sets)

    groups = list(sets)
    for i in range(2, max_order + 1):
        for comb in combinations(sets, i):
            groups.append(
                comb[0].intersection(*comb[1:]))
    return groups
