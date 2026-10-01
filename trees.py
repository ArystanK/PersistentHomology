import itertools
from dataclasses import dataclass, field
from math import cos
from typing import Callable, FrozenSet


@dataclass(frozen=True)
class Node[T]:
    value: T | None = None
    children: FrozenSet["Node[T]"] = field(default_factory=frozenset)


def is_tree_isomorphic[T](
        tree1: Node[T],
        tree2: Node[T],
        vertex_predicate: Callable[[T | None, T | None], bool] = lambda a, b: True
) -> bool:
    if not vertex_predicate(tree1.value, tree2.value):
        return False

    if len(tree1.children) != len(tree2.children):
        return False

    if not tree1.children:
        return True

    remaining_children2 = list(tree2.children)

    for child1 in tree1.children:
        match_found = False
        for child2 in remaining_children2:
            if is_tree_isomorphic(child1, child2, vertex_predicate):
                remaining_children2.remove(child2)
                match_found = True
                break

        if not match_found:
            return False

    return True


type Graph[T] = dict[T, FrozenSet[T]]


def is_graph_isomorphic[T](
        graph1: Graph[T],
        graph2: Graph[T],
        vertex_predicate: Callable[[T | None, T | None], bool] = lambda a, b: True
) -> bool:
    """
    Determines if graph1 and graph2 are isomorphic under a vertex predicate constraint.
    """
    nodes1 = list(graph1.keys())
    nodes2 = list(graph2.keys())

    # Check structural prerequisite: node counts must match
    if len(nodes1) != len(nodes2):
        return False

    # Check empty graphs scenario
    if not nodes1:
        return True

    # Check every possible bijective vertex mapping permutation
    for p in itertools.permutations(nodes2):
        mapping = dict(zip(nodes1, p))

        # 1. Enforce the node attribute constraints via the predicate
        if not all(vertex_predicate(n1, mapping[n1]) for n1 in nodes1):
            continue

        # 2. Check edge preservation (Bijection of adjacency sets)
        isomorphism_found = True
        for n1 in nodes1:
            # Map neighbors from graph1 to their mapped equivalents in graph2
            mapped_neighbors = frozenset(mapping[nbr] for nbr in graph1[n1])

            # The mapped neighbors must match the target node's actual neighbors
            if mapped_neighbors != graph2[mapping[n1]]:
                isomorphism_found = False
                break

        if isomorphism_found:
            return True

    return False


def main():
    tree1 = Node(
        value=1,
        children=frozenset([Node(value=2), Node(value=3), Node(value=4)]),
    )
    tree2 = Node(
        value=1,
        children=frozenset([Node(value=2), Node(value=3), Node(value=4)])
    )
    print(is_tree_isomorphic(tree1, tree2, vertex_predicate=lambda a, b: a == b))


if __name__ == "__main__":
    x = 0
    end = 1
    eps = 1 / 2_000_000
    delta = 1 / 100_000_000
    count = 0
    while x <= end:
        if -eps <= cos(97 * x) - x <= eps:
            count += 1
        x += delta

    print(count)
