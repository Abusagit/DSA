import sys
from typing import TypeVar, Callable, DefaultDict, Generator, Set, Tuple, List, Generic
from collections import defaultdict

KT = TypeVar('KT')
VT = TypeVar('VT')
T = TypeVar('T')


def identity(x: T) -> T:
    return x


class ArgDefaultDict(DefaultDict[KT, VT]):
    def __init__(self, fun_with_param: Callable[[KT], VT]):
        super().__init__(None)
        self.fun: Callable[[KT], VT] = fun_with_param

    def __missing__(self, key: KT) -> VT:
        ret: VT = self.fun(key)
        self[key] = ret
        return ret


class DisjointSet(Generic[T]):
    def __init__(self) -> None:
        self._data: ArgDefaultDict[T, T] = ArgDefaultDict(identity)

    def __contains__(self, item: T) -> bool:
        return item in self._data

    def __bool__(self) -> bool:
        return bool(self._data)

    def __repr__(self) -> str:
        value_dict: DefaultDict[T, List[T]] = defaultdict(list)
        for key, value in sorted(self._data.items()):
            value_dict[value].append(key)
        return "{classname}({values})".format(
            classname=self.__class__.__name__,
            values=', '.join([f'{key} <- {value}' for key, value in value_dict.items()]),
        )

    def __iter__(self) -> Generator[Tuple[T, T], None, None]:
        for key in self._data:
            yield key, self.find(key)

    def itersets(self) -> Generator[Set[T], None, None]:
        """
        Yields sets of connected components.
        >>> ds = DisjointSet()
        >>> ds.union(1,2)
        >>> list(ds.itersets())
        [{1, 2}]
        """
        element_classes: DefaultDict[T, Set[T]] = defaultdict(lambda: set())
        for element in self._data:
            element_classes[self.find(element)].add(element)

        for element_class in element_classes.values():
            yield element_class

    def find(self, x: T) -> T:
        """
        Returns the representative member of the set of connected components to which x belongs, may be x itself.
        >>> ds = DisjointSet()
        >>> ds.find(1)
        1
        >>> ds.union(1,2)
        >>> ds.find(1)
        2
        """
        if x != self._data[x]:
            self._data[x] = self.find(self._data[x])
        return self._data[x]

    def union(self, x: T, y: T) -> None:
        """
        Attaches the roots of x and y trees together if they are not the same already.
        :param x: first element
        :param y: second element
        :return: None
        """
        parent_x, parent_y = self.find(x), self.find(y)
        if parent_x != parent_y:
            self._data[parent_x] = parent_y

    def connected(self, x: T, y: T) -> bool:
        """
        :param x: first element
        :param y: second element
        :return: True if x and y belong to the same tree (i.e. they have the same root), False otherwise.
        >>> ds = DisjointSet()
        >>> ds.connected(1,2)
        False
        >>> ds.union(1,2)
        >>> ds.connected(1,2)
        True
        """
        return self.find(x) == self.find(y)


class DisjointSet2:
    def __init__(self, elements):
        self.number = len(elements)
        self.elements = elements
        self.parent = {i: i for i in range(self.number)}
        self.rank = {i: 0 for i in range(self.number)}

    def find(self, index):
        if self.parent[index] != index:
            self.parent[index] = self.find(self.parent[index])
        return self.parent[index]

    def union(self, i, j):
        i_id = self.find(i)
        j_id = self.find(j)
        if i_id == j_id:
            return
        if self.rank[i_id] > self.rank[j_id]:
            self.parent[j_id] = i_id
        else:
            self.parent[i_id] = j_id
            if self.rank[i_id] == self.rank[j_id]:
                self.rank[j_id] += 1
