from dataclasses import dataclass
from itertools import combinations
from typing import ClassVar, DefaultDict, Dict, List, Set, Iterable, Union, TypeVar, Protocol, runtime_checkable


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

    def __len__(self):
        return self.cardinality


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

    def difference(self, *others: Iterable[Union[Set[T], List[T], SetLike[T]]]) -> 'SetDifference':
        return SetDifference(SetDifference.make_name([self, *others]), [self, *others])

    def __and__(self, other: SetLike):
        return self.intersection(other)

    def __or__(self, other: SetLike):
        return self.union(other)

    def __sub__(self, other: SetLike):
        return self.difference(other)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name!r}, {self.elements})"

    degree: ClassVar[int] = 1


@dataclass(repr=False)
class VSet(BaseSet[T]):

    _elements: Set[T]

    group_type: ClassVar[str] = 'set'

    def get_elements(self) -> Set[T]:
        return self._elements

    def is_of(self, setlike: BaseSet[T]) -> bool:
        return self.name == setlike.name

    def component_overlaps(self, other: BaseSet[T]) -> Set[str]:
        if isinstance(other, SetCombination):
            return {self.name} & {s.name for s in other.sets}
        else:
            return {self.name} if self.name == other.name else set()

@dataclass(repr=False)
class SetCombination(BaseSet[T]):
    sets: List[SetLike[T]]
    _symbol = "?"

    def is_of(self, setlike: BaseSet[T]) -> bool:
        for member in self.sets:
            if member.name == setlike.name:
                return True
        return False

    def component_overlaps(self, other: BaseSet[T]) -> Set[str]:
        if isinstance(other, SetCombination):
            return {s.name for s in self.sets} & {s.name for s in other.sets}
        else:
            return {other.name} if any(other.name == s.name for s in self.sets) else set()


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


@dataclass(repr=False)
class SetDifference(SetCombination[T]):
    group_type: ClassVar[str] = 'difference'
    _symbol = '-'

    def get_elements(self) -> Set[T]:
        base = set(self.sets[0].get_elements())
        for s in self.sets[1:]:
            base -= s.get_elements()
        return base


def combinate_sets(sets: List[VSet[T]]) -> List[SetLike[T]]:
    max_order = len(sets)

    groups = list(sets)
    for i in range(2, max_order + 1):
        for comb in combinations(sets, i):
            groups.append(
                comb[0].intersection(*comb[1:]))
    return groups


def compute_exclusive_sizes(sets: List[BaseSet[T]]) -> Dict[str, int]:
    by_size = DefaultDict(list)
    for c in sets:
        by_size[c.degree].append(c)

    exclusive_sizes = {}

    z = max(by_size)
    for group in by_size[z]:
        exclusive_sizes[group.name] = group.cardinality

    for i in range(z - 1, 0, -1):
        prev = by_size[i + 1]
        for group in by_size[i]:
            spanned_by = group.difference(*prev)
            exclusive_sizes[group.name] = spanned_by.cardinality
    return exclusive_sizes
