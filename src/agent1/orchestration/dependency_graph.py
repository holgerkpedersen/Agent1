"""Dependency graph management and analysis utilities.

This module provides classes and functions to build, analyze and visualize the
dependency relationships between tasks within a orchestration pipeline.  It is
designed to be used by :mod:`src.agent1.orchestration` but can also be imported
stand-alone for testing or CLI tooling.

The graph is implemented on top of :class:`networkx.DiGraph` so that we get all
the standard algorithms (topological sort, cycle detection, transitive
closure, ...) for free while keeping the API small and focused on our domain.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Optional, Set, Tuple

try:  # pragma: no cover - networkx is an optional dependency in some envs
    import networkx as nx
except ImportError:  # type: ignore[assignment]
    nx = None  # noqa: N816 - we keep the alias short for readability


# ---------------------------------------------------------------------------
# Graph backend protocol (shared by networkx.DiGraph and _FallbackDiGraph)
# ---------------------------------------------------------------------------
class GraphBackend(Protocol):
    """Common interface satisfied by both the networkx backend and fallback.

    Declaring this lets mypy verify ``DependencyGraph`` against a single typed
    surface instead of losing method typing when ``nx`` is absent, removing the
    cluster of ``type: ignore[union-attr/index/operator]`` markers throughout the
    class body.  Both backends implement every member below with identical semantics.
    """

    def add_node(self, node_id: str, **attrs: object) -> None: ...
    def add_edge(self, u: str, v: str) -> None: ...
    def has_node(self, node_id: str) -> bool: ...
    @property
    def nodes(self) -> Dict[str, Dict[str, object]]: ...
    @property
    def edges(self) -> List[Tuple[str, str]]: ...
    def successors(self, node_id: str) -> Iterator[str]: ...
    def predecessors(self, node_id: str) -> Iterator[str]: ...
    def number_of_nodes(self) -> int: ...
    def number_of_edges(self) -> int: ...
    def in_degree(self, node_id: str) -> int: ...
    def out_degree(self, node_id: str) -> int: ...


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class DependencyGraphError(Exception):
    """Base exception for all dependency-graph related errors."""


class CycleError(DependencyGraphError):
    """Raised when a cycle is detected in the dependency graph.

    The ``cycle`` attribute holds one representative cycle as a list of node
    identifiers so callers can surface it to users or logs.
    """

    def __init__(self, message: str, cycle: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.cycle: List[str] = cycle or []


class UnknownNodeError(DependencyGraphError):
    """Raised when an operation references a node that does not exist."""

    def __init__(self, node_id: str) -> None:
        super().__init__(f"Unknown node '{node_id}'")
        self.node_id = node_id


# ---------------------------------------------------------------------------
# Core data model
# ---------------------------------------------------------------------------
class TaskNode:
    """A lightweight container describing a single task in the graph.

    Attributes are intentionally kept minimal; richer metadata can be attached
    via :meth:`DependencyGraph.set_metadata`.
    """

    __slots__ = ("node_id", "task_type", "priority", "metadata")

    def __init__(
        self,
        node_id: str,
        task_type: Optional[str] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, object]] = None,
    ) -> None:
        if not node_id:
            raise ValueError("node_id must be a non-empty string")
        self.node_id: str = node_id
        self.task_type: Optional[str] = task_type
        self.priority: int = priority
        self.metadata: Dict[str, object] = dict(metadata) if metadata else {}

    # -- helpers -----------------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        return {
            "node_id": self.node_id,
            "task_type": self.task_type,
            "priority": self.priority,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "TaskNode":
        return cls(
            node_id=str(data["node_id"]),
            task_type=data.get("task_type"),  # type: ignore[arg-type]
            priority=int(data.get("priority", 0)),  # type: ignore[arg-type]
            metadata=dict(data.get("metadata", {})),  # type: ignore[arg-type]
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"TaskNode(node_id={self.node_id!r}, task_type={self.task_type!r})"


# ---------------------------------------------------------------------------
# DependencyGraph
# ---------------------------------------------------------------------------
class DependencyGraph:
    """Directed acyclic graph (DAG) of task dependencies.

    The class wraps :class:`networkx.DiGraph` when available and falls back to
    a pure-Python implementation otherwise so that the module remains usable in
    restricted environments.  All public methods have identical semantics
    regardless of backend.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name: str = name or "default"
        if nx is not None:
            self._graph: GraphBackend = nx.DiGraph()  # type: ignore[assignment]
        else:
            self._graph = _FallbackDiGraph()

    # -- construction ------------------------------------------------------
    def add_node(
        self,
        node_id: str,
        task_type: Optional[str] = None,
        priority: int = 0,
        metadata: Optional[Dict[str, object]] = None,
    ) -> TaskNode:
        """Add a task node to the graph.

        Returns the :class:`TaskNode` instance that was registered so callers
        can keep a reference without re-looking-up by id.
        """
        if self.has_node(node_id):
            raise DependencyGraphError(f"Node '{node_id}' already exists")
        node = TaskNode(
            node_id=node_id, task_type=task_type, priority=priority, metadata=metadata
        )
        self._graph.add_node(node_id, _task_node=node)
        return node

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Register that ``to_node`` depends on ``from_node``.

        In other words data flows *from* ``from_node`` *to* ``to_node``.  Both
        endpoints must already exist as nodes.
        """
        for nid in (from_node, to_node):
            if not self.has_node(nid):
                raise UnknownNodeError(nid)
        if from_node == to_node:
            raise DependencyGraphError("Self-loops are not permitted")
        self._graph.add_edge(from_node, to_node)

    def add_dependency(self, dependent: str, dependency: str) -> None:
        """Alias for :meth:`add_edge` using a more readable argument order.

        ``dependent`` is the task that requires ``dependency``.
        """
        self.add_edge(dependency, dependent)

    # -- queries -----------------------------------------------------------
    def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    def get_node(self, node_id: str) -> TaskNode:
        if not self.has_node(node_id):
            raise UnknownNodeError(node_id)
        data = self._graph.nodes[node_id]
        return data["_task_node"]

    def neighbors(self, node_id: str) -> List[str]:
        """Return the immediate successors (dependents) of ``node_id``."""
        if not self.has_node(node_id):
            raise UnknownNodeError(node_id)
        return list(self._graph.successors(node_id))

    def predecessors(self, node_id: str) -> List[str]:
        """Return the immediate dependencies (sources) of ``node_id``."""
        if not self.has_node(node_id):
            raise UnknownNodeError(node_id)
        return list(self._graph.predecessors(node_id))

    def all_nodes(self) -> List[str]:
        return list(self._graph.nodes)

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    # -- metadata ----------------------------------------------------------
    def set_metadata(self, node_id: str, key: str, value: object) -> None:
        if not self.has_node(node_id):
            raise UnknownNodeError(node_id)
        node = self.get_node(node_id)
        node.metadata[key] = value

    def get_metadata(self, node_id: str, key: str) -> Optional[object]:
        node = self.get_node(node_id)
        return node.metadata.get(key)

    # -- analysis ----------------------------------------------------------
    def topological_order(self) -> List[str]:
        """Return nodes in an order that respects all dependencies.

        Raises :class:`CycleError` if the graph contains a cycle.
        """
        if nx is not None:
            try:
                return list(nx.topological_sort(self._graph))
            except nx.NetworkXUnfeasible as exc:  # pragma: no cover - backend
                raise CycleError(str(exc), cycle=self.find_cycle()) from exc
        order, _ = self._graph.topological_sort()
        return order

    def find_cycle(self) -> List[str]:
        """Return one representative cycle as a list of node ids.

        Returns an empty list when the graph is acyclic.
        """
        if nx is not None:
            try:
                edges = nx.find_cycle(self._graph, orientation="original")
                return [edges[0][0]] + [e[1] for e in edges]
            except nx.NetworkXNoCycle:
                return []
        cycle = self._graph.find_cycle()
        return list(cycle)

    def transitive_closure(self, node_id: str) -> Set[str]:
        """All nodes reachable from ``node_id`` (including itself)."""
        if not self.has_node(node_id):
            raise UnknownNodeError(node_id)
        if nx is not None:
            return set(nx.descendants(self._graph, node_id)) | {node_id}  # type: ignore[operator]
        reachable = self._graph.reachable_from(node_id)
        return set(reachable)

    def ancestors(self, node_id: str) -> Set[str]:
        """All nodes from which ``node_id`` is ultimately dependent."""
        if not self.has_node(node_id):
            raise UnknownNodeError(node_id)
        if nx is not None:
            return set(nx.ancestors(self._graph, node_id))  # type: ignore[operator]
        return set(self._graph.ancestors_of(node_id))  # type: ignore[operator]

    def descendants(self, node_id: str) -> Set[str]:
        """All nodes that depend (directly or transitively) on ``node_id``."""
        if not self.has_node(node_id):
            raise UnknownNodeError(node_id)
        return self.transitive_closure(node_id) - {node_id}

    def critical_path(self) -> List[str]:
        """Longest dependency chain in the graph.

        Uses node priorities as weights when available; otherwise counts edges.
        Raises :class:`CycleError` if a cycle exists.
        """
        if self.find_cycle():
            raise CycleError("Graph contains a cycle", cycle=self.find_cycle())
        order = self.topological_order()
        # weight lookup helper -------------------------------------------------
        def _weight(u: str, v: str) -> int:
            node_v = self.get_node(v)
            return node_v.priority

        if nx is not None:
            dag = self._graph  # type: ignore[assignment]
            longest = nx.dag_longest_path(dag, weight=_weight)  # type: ignore[arg-type]
            return list(longest)
        return self._graph.longest_path(weight=_weight)

    def roots(self) -> List[str]:
        """Nodes with no dependencies."""
        if nx is not None:
            return [n for n in self._graph.nodes if self._graph.in_degree(n) == 0]  # type: ignore[operator]
        return list(self._graph.roots())

    def leaves(self) -> List[str]:
        """Nodes that nothing else depends on."""
        if nx is not None:
            return [n for n in self._graph.nodes if self._graph.out_degree(n) == 0]  # type: ignore[operator]
        return list(self._graph.leaves())

    def validate_acyclic(self) -> bool:
        """Return ``True`` when the graph is acyclic, else raise."""
        if self.find_cycle():
            raise CycleError("Graph contains a cycle", cycle=self.find_cycle())
        return True

    # -- serialization -----------------------------------------------------
    def to_dict(self) -> Dict[str, object]:
        nodes: List[Dict[str, object]] = []
        edges: List[List[str]] = []
        for nid in self.all_nodes():
            node = self.get_node(nid)
            nodes.append(node.to_dict())
        if nx is not None:
            edge_iter: Iterator[Tuple[str, str]] = iter(self._graph.edges)
        else:
            edge_iter = iter(self._graph.edges)
        for u, v in edge_iter:
            edges.append([u, v])
        return {"name": self.name, "nodes": nodes, "edges": edges}

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "DependencyGraph":
        graph = cls(name=str(data.get("name", "default")))  # type: ignore[arg-type]
        for node_data in data.get("nodes", []):  # type: ignore[union-attr]
            graph.add_node(
                **TaskNode.from_dict(node_data).to_dict()  # type: ignore[arg-type]
            )
        for edge_pair in data.get("edges", []):  # type: ignore[union-attr]
            u, v = edge_pair  # type: ignore[misc]
            graph.add_edge(u, v)
        return graph

    @classmethod
    def from_json(cls, payload: str) -> "DependencyGraph":
        return cls.from_dict(json.loads(payload))


# ---------------------------------------------------------------------------
# Fallback pure-Python implementation (used when networkx is unavailable)
# ---------------------------------------------------------------------------
class _FallbackDiGraph:
    """Minimal directed graph supporting the subset of operations we need."""

    def __init__(self) -> None:
        self._nodes: Dict[str, Dict[str, object]] = {}
        self._succ: Dict[str, Set[str]] = defaultdict(set)
        self._pred: Dict[str, Set[str]] = defaultdict(set)
        self._edges: List[Tuple[str, str]] = []

    # -- mutators ----------------------------------------------------------
    def add_node(self, node_id: str, **attrs: object) -> None:
        if node_id not in self._nodes:
            self._nodes[node_id] = dict(attrs)
            self._succ[node_id]  # noqa: B018 - touch defaultdict
            self._pred[node_id]

    def add_edge(self, u: str, v: str) -> None:
        if u not in self._nodes or v not in self._nodes:
            raise UnknownNodeError(u if u not in self._nodes else v)
        self._succ[u].add(v)
        self._pred[v].add(u)
        self._edges.append((u, v))

    # -- accessors ---------------------------------------------------------
    def has_node(self, node_id: str) -> bool:
        return node_id in self._nodes

    @property
    def nodes(self) -> Dict[str, Dict[str, object]]:
        return self._nodes

    @property
    def edges(self) -> List[Tuple[str, str]]:
        return list(self._edges)

    def successors(self, node_id: str) -> Iterator[str]:
        if not self.has_node(node_id):
            raise UnknownNodeError(node_id)
        yield from sorted(self._succ[node_id])

    def predecessors(self, node_id: str) -> Iterator[str]:
        if not self.has_node(node_id):
            raise UnknownNodeError(node_id)
        yield from sorted(self._pred[node_id])

    # -- algorithms --------------------------------------------------------
    def topological_sort(self) -> Tuple[List[str], Dict[str, int]]:
        """Kahn's algorithm.  Returns (order, position_map)."""
        in_degree: Dict[str, int] = {n: len(self._pred[n]) for n in self._nodes}
        queue: List[str] = sorted(n for n, d in in_degree.items() if d == 0)
        order: List[str] = []
        while queue:
            node = queue.pop(0)
            order.append(node)
            for succ in sorted(self._succ[node]):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
            # keep deterministic ordering
            queue.sort()
        if len(order) != len(self._nodes):
            raise CycleError("Cycle detected", cycle=self.find_cycle())
        position = {n: i for i, n in enumerate(order)}
        return order, position

    def find_cycle(self) -> List[str]:
        """DFS-based cycle search returning one representative cycle."""
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {n: WHITE for n in self._nodes}
        parent: Dict[str, Optional[str]] = {n: None for n in self._nodes}

        def _dfs(start: str) -> List[str]:
            stack: List[Tuple[str, Iterator[str]]] = [(start, iter(sorted(self._succ[start])))]
            color[start] = GRAY
            while stack:
                node, it = stack[-1]
                advanced = False
                for nxt in it:
                    if color[nxt] == WHITE:
                        color[nxt] = GRAY
                        parent[nxt] = node
                        stack.append((nxt, iter(sorted(self._succ[nxt]))))
                        advanced = True
                        break
                    elif color[nxt] == GRAY:
                        # reconstruct cycle from nxt back to itself via parents
                        cycle: List[str] = [nxt]
                        cur = node
                        while cur is not None and cur != nxt:
                            cycle.append(cur)
                            cur = parent[cur]  # type: ignore[assignment]
                        cycle.reverse()
                        cycle.append(nxt)
                        return cycle
                if not advanced:
                    color[node] = BLACK
                    stack.pop()
            return []

        for n in sorted(self._nodes):
            if color[n] == WHITE:
                cyc = _dfs(n)
                if cyc:
                    return cyc
        return []

    def reachable_from(self, node_id: str) -> Set[str]:
        seen: Set[str] = {node_id}
        frontier: List[str] = [node_id]
        while frontier:
            nxt_frontier: List[str] = []
            for n in frontier:
                for s in self._succ[n]:
                    if s not in seen:
                        seen.add(s)
                        nxt_frontier.append(s)
            frontier = nxt_frontier
        return seen

    def ancestors_of(self, node_id: str) -> Set[str]:
        seen: Set[str] = {node_id}
        frontier: List[str] = [node_id]
        while frontier:
            nxt_frontier: List[str] = []
            for n in frontier:
                for p in self._pred[n]:
                    if p not in seen:
                        seen.add(p)
                        nxt_frontier.append(p)
            frontier = nxt_frontier
        return seen

    def longest_path(self, weight=None) -> List[str]:
        order, _ = self.topological_sort()
        best: Dict[str, Tuple[int, Optional[str]]] = {n: (0 if not weight else 0, None) for n in self._nodes}
        # seed with priority of each node as its own length contribution
        for n in order:
            base = best[n][0]
            w_self = weight(n, n) if weight else 0  # noqa: E731 - simple lambda-free call
            # We treat the path length as sum of successor weights.
            for s in self._succ[n]:
                edge_weight = (weight(n, s) or 0) if weight else 1
                candidate = base + edge_weight
                if candidate > best[s][0]:
                    best[s] = (candidate, n)
        # find terminal with highest score
        end = max(order, key=lambda x: best[x][0])  # noqa: E731 - clarity over brevity
        path: List[str] = [end]
        cur = best[end][1]
        while cur is not None:
            path.append(cur)
            cur = best[cur][1]
        path.reverse()
        return path

    def roots(self) -> List[str]:
        return sorted(n for n in self._nodes if not self._pred[n])

    def leaves(self) -> List[str]:
        return sorted(n for n in self._nodes if not self._succ[n])


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------
def build_graph_from_pairs(
    pairs: Iterable[Tuple[str, str]], *, name: Optional[str] = None
) -> DependencyGraph:
    """Quickly create a graph from an iterable of ``(dependent, dependency)`` tuples.

    Nodes are auto-created in the order they appear so callers don't need to
    pre-register them.  Duplicate edges and self-loops are silently ignored.
    """
    graph = DependencyGraph(name=name)
    seen_nodes: Set[str] = set()
    for dependent, dependency in pairs:
        if dependent == dependency:
            continue
        for nid in (dependent, dependency):
            if nid not in seen_nodes and not graph.has_node(nid):
                graph.add_node(nid)
                seen_nodes.add(nid)
        try:
            graph.add_edge(dependency, dependent)
        except DependencyGraphError:
            # duplicate edge - ignore
            continue
    return graph


def merge_graphs(*graphs: DependencyGraph) -> DependencyGraph:
    """Combine multiple graphs into one.

    Raises if any constituent graph has a cycle or if node ids collide.
    """
    merged = DependencyGraph(name="merged")
    for g in graphs:
        g.validate_acyclic()
        for nid in g.all_nodes():
            if merged.has_node(nid):
                raise DependencyGraphError(f"Duplicate node '{nid}' during merge")
            node = g.get_node(nid)
            merged.add_node(
                node_id=node.node_id,
                task_type=node.task_type,
                priority=node.priority,
                metadata=dict(node.metadata),
            )
        if nx is not None:
            edge_iter: Iterator[Tuple[str, str]] = iter(g._graph.edges)
        else:
            edge_iter = iter(g._graph.edges)
        for u, v in edge_iter:
            merged.add_edge(u, v)
    return merged


__all__: List[str] = [
    "DependencyGraph",
    "TaskNode",
    "DependencyGraphError",
    "CycleError",
    "UnknownNodeError",
    "build_graph_from_pairs",
    "merge_graphs",
]