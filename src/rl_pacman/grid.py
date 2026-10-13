"""Static maze geometry: walls, movement, tunnels and BFS distances."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import IntEnum

Pos = tuple[int, int]  # (row, col)


class Action(IntEnum):
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


DELTAS: dict[Action, Pos] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


@dataclass(frozen=True)
class Grid:
    """Immutable maze parsed from a layout string."""

    height: int
    width: int
    walls: frozenset[Pos]
    dots: frozenset[Pos]
    pellets: frozenset[Pos]
    pacman_start: Pos
    ghost_starts: tuple[Pos, ...]

    @classmethod
    def from_layout(cls, layout: str) -> Grid:
        rows = layout.splitlines()
        if not rows:
            raise ValueError("empty layout")
        width = max(len(r) for r in rows)
        rows = [r.ljust(width) for r in rows]

        walls: set[Pos] = set()
        dots: set[Pos] = set()
        pellets: set[Pos] = set()
        pacman: list[Pos] = []
        ghosts: list[Pos] = []
        symbols = {
            "#": walls,
            ".": dots,
            "o": pellets,
        }
        for r, row in enumerate(rows):
            for c, cell in enumerate(row):
                if cell in symbols:
                    symbols[cell].add((r, c))
                elif cell == "P":
                    pacman.append((r, c))
                elif cell == "G":
                    ghosts.append((r, c))
                elif cell != " ":
                    raise ValueError(f"unknown cell {cell!r} at {(r, c)}")

        if len(pacman) != 1:
            raise ValueError(f"expected exactly one 'P', got {len(pacman)}")
        return cls(
            height=len(rows),
            width=width,
            walls=frozenset(walls),
            dots=frozenset(dots),
            pellets=frozenset(pellets),
            pacman_start=pacman[0],
            ghost_starts=tuple(ghosts),
        )

    def move(self, pos: Pos, action: Action) -> Pos:
        """Cell reached from `pos` with `action`; `pos` itself if blocked.

        Leaving the grid wraps around to the other side (tunnels).
        """
        dr, dc = DELTAS[action]
        r = (pos[0] + dr) % self.height
        c = (pos[1] + dc) % self.width
        if (r, c) in self.walls:
            return pos
        return (r, c)

    def legal_actions(self, pos: Pos) -> list[Action]:
        """Actions that actually move from `pos`."""
        return [a for a in Action if self.move(pos, a) != pos]

    def distances_from(self, source: Pos) -> dict[Pos, int]:
        """BFS distance (in moves) from `source` to every reachable cell."""
        dist = {source: 0}
        queue = deque([source])
        while queue:
            cur = queue.popleft()
            for action in Action:
                nxt = self.move(cur, action)
                if nxt not in dist:
                    dist[nxt] = dist[cur] + 1
                    queue.append(nxt)
        return dist
