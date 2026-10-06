"""Knowledge graph traversal with explicit inverse relation IDs."""

import json
from collections import deque
from typing import Dict, List, Tuple


class KGTriple:
    def __init__(self, head: int, relation: int, tail: int):
        self.head = head
        self.relation = relation
        self.tail = tail


class KnowledgeGraph:
    def __init__(self, entity2id: Dict, relation2id: Dict, triples: List[KGTriple]):
        if sorted(entity2id.values()) != list(range(len(entity2id))):
            raise ValueError("Entity IDs must be contiguous from zero")
        if sorted(relation2id.values()) != list(range(len(relation2id))):
            raise ValueError("Relation IDs must be contiguous from zero")
        self.entity2id = entity2id
        self.relation2id = relation2id
        self.id2entity = {v: k for k, v in entity2id.items()}
        base_relations = len(relation2id)
        self.id2relation = {v: k for k, v in relation2id.items()}
        self.id2relation.update({
            v + base_relations: f"inverse_of_{k}"
            for k, v in relation2id.items()
        })
        self.triples = triples
        self.n_entities = len(entity2id)
        self.n_relations = 2 * base_relations

        self.adj: Dict[int, List[Tuple[int, int]]] = {}
        for t in triples:
            if not (0 <= t.head < self.n_entities and 0 <= t.tail < self.n_entities):
                raise ValueError("KG triple refers to an unknown entity")
            if not 0 <= t.relation < base_relations:
                raise ValueError("KG triple refers to an unknown relation")
            self.adj.setdefault(t.head, []).append((t.relation, t.tail))
            self.adj.setdefault(t.tail, []).append(
                (t.relation + base_relations, t.head)
            )

        self.degree = {node: len(neighbors) for node, neighbors in self.adj.items()}

    def get_neighbors(self, node: int) -> List[Tuple[int, int]]:
        return self.adj.get(node, [])

    def get_degree(self, node: int) -> int:
        return self.degree.get(node, 0)

    def multi_hop_paths(
        self,
        src: int,
        dst: int,
        max_hop: int = 3,
        max_paths: int = 200,
        max_neighbors: int = 50,
    ) -> List[Dict]:
        paths = []

        def dfs(node: int, path: List[Tuple[int, int, int]], visited: set):
            if len(paths) >= max_paths:
                return

            if len(path) > 0 and node == dst:
                paths.append({
                    "path": path[:],
                    "hop": len(path),
                })
                return

            if len(path) >= max_hop:
                return

            neighbors = self.get_neighbors(node)
            if len(neighbors) > max_neighbors:
                neighbors = sorted(
                    neighbors,
                    key=lambda entry: (
                        entry[1] != dst, self.get_degree(entry[1]), entry[1]
                    ),
                )[:max_neighbors]

            for rel, neighbor in neighbors:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append((node, rel, neighbor))
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.discard(neighbor)

        dfs(src, [], {src})
        return paths

    def paths_around_target(
        self, target: int, max_hop: int = 3,
        max_paths: int = 200, max_neighbors: int = 50,
    ) -> List[Dict]:
        """Enumerate a bounded target neighborhood once for offline indexing."""
        paths = []
        inverse_offset = self.n_relations // 2

        def inverse(relation):
            return (
                relation + inverse_offset
                if relation < inverse_offset else relation - inverse_offset
            )

        quota = max(1, max_paths // max_hop)
        depth_counts = [0] * (max_hop + 1)
        frontier = deque([(target, [], {target})])
        while frontier and len(paths) < max_paths:
            node, reverse_path, visited = frontier.popleft()
            if len(reverse_path) >= max_hop:
                continue
            neighbors = sorted(
                self.get_neighbors(node),
                key=lambda entry: (self.get_degree(entry[1]), entry[1]),
            )[:max_neighbors]
            for relation, neighbor in neighbors:
                if neighbor in visited:
                    continue
                forward_path = [(neighbor, inverse(relation), node), *reverse_path]
                depth = len(forward_path)
                if depth_counts[depth] >= quota:
                    break
                paths.append({"path": forward_path, "hop": len(forward_path)})
                depth_counts[depth] += 1
                if len(paths) >= max_paths:
                    break
                frontier.append((neighbor, forward_path, visited | {neighbor}))
        return paths

    @classmethod
    def from_files(cls, entity_file: str, relation_file: str, triple_file: str):
        with open(entity_file) as f:
            entity2id = json.load(f)
        with open(relation_file) as f:
            relation2id = json.load(f)
        with open(triple_file) as f:
            raw_triples = json.load(f)
        triples = [KGTriple(t[0], t[1], t[2]) for t in raw_triples]
        return cls(entity2id, relation2id, triples)
