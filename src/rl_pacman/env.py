"""Pacman as an RL environment, with no dependency on pygame.

The interface follows the usual `reset() -> state`,
`step(action) -> StepResult` convention, so any agent can be plugged in
and the game can run headless (much faster than real time) for training.
"""

from __future__ import annotations

import random
from collections.abc import Hashable
from dataclasses import dataclass, field

from rl_pacman.grid import Action, Grid, Pos
from rl_pacman.layouts import LAYOUTS

__all__ = [
    "Action",
    "EnvConfig",
    "PacmanEnv",
    "Rewards",
    "State",
    "StepResult",
]


@dataclass(frozen=True)
class Rewards:
    step: float = -1.0
    dot: float = 10.0
    pellet: float = 50.0
    eat_ghost: float = 200.0
    win: float = 500.0
    death: float = -500.0


@dataclass(frozen=True)
class EnvConfig:
    layout: str = "small"
    # Ghosts move once every `ghost_period` Pacman steps (1 = same speed).
    ghost_period: int = 2
    # Probability that a ghost takes a random legal move instead of chasing.
    ghost_random_prob: float = 0.2
    frightened_steps: int = 20
    max_steps: int = 300
    rewards: Rewards = field(default_factory=Rewards)


@dataclass(frozen=True)
class State:
    """Full, hashable game state."""

    pacman: Pos
    ghosts: tuple[Pos, ...]
    dots: frozenset[Pos]
    pellets: frozenset[Pos]
    frightened: int  # remaining frightened steps, 0 = ghosts are dangerous

    def key(self) -> Hashable:
        """Compact key for a tabular agent (Q-table index).

        The exact frightened counter is collapsed into a boolean to keep
        the number of states small.
        """
        return (
            self.pacman,
            self.ghosts,
            self.dots,
            self.pellets,
            self.frightened > 0,
        )


@dataclass(frozen=True)
class StepResult:
    state: State
    reward: float
    done: bool
    # "win", "death", "timeout" or "" while the episode is running.
    outcome: str = ""


class PacmanEnv:
    def __init__(
        self, config: EnvConfig | None = None, seed: int | None = None
    ) -> None:
        self.config = config or EnvConfig()
        layout = LAYOUTS.get(self.config.layout, self.config.layout)
        self.grid = Grid.from_layout(layout)
        self.rng = random.Random(seed)
        self.state = self._initial_state()
        self.steps = 0
        self.done = False
        self.score = 0.0

    @property
    def n_actions(self) -> int:
        return len(Action)

    def _initial_state(self) -> State:
        return State(
            pacman=self.grid.pacman_start,
            ghosts=self.grid.ghost_starts,
            dots=self.grid.dots,
            pellets=self.grid.pellets,
            frightened=0,
        )

    def reset(self) -> State:
        self.state = self._initial_state()
        self.steps = 0
        self.done = False
        self.score = 0.0
        return self.state

    def step(self, action: Action | int) -> StepResult:
        if self.done:
            raise RuntimeError("episode is over, call reset()")
        action = Action(action)
        rw = self.config.rewards
        s = self.state
        self.steps += 1
        reward = rw.step

        # 1. Pacman moves and eats.
        pacman = self.grid.move(s.pacman, action)
        dots, pellets, frightened = s.dots, s.pellets, s.frightened
        if pacman in dots:
            dots = dots - {pacman}
            reward += rw.dot
        elif pacman in pellets:
            pellets = pellets - {pacman}
            reward += rw.pellet
            frightened = self.config.frightened_steps + 1

        # 2. Collision before ghosts move (Pacman walked into a ghost).
        ghosts, gained, dead = self._collide(pacman, s.ghosts, frightened)
        reward += gained

        # 3. Ghosts move, then collision again (a ghost walked into Pacman).
        if not dead and self.steps % self.config.ghost_period == 0:
            ghosts = self._move_ghosts(pacman, ghosts, frightened)
            ghosts, gained, dead = self._collide(pacman, ghosts, frightened)
            reward += gained

        frightened = max(0, frightened - 1)
        self.state = State(pacman, ghosts, dots, pellets, frightened)

        outcome = ""
        if dead:
            reward += rw.death
            outcome = "death"
        elif not dots and not pellets:
            reward += rw.win
            outcome = "win"
        elif self.steps >= self.config.max_steps:
            outcome = "timeout"

        self.done = outcome != ""
        self.score += reward
        return StepResult(self.state, reward, self.done, outcome)

    def _collide(
        self, pacman: Pos, ghosts: tuple[Pos, ...], frightened: int
    ) -> tuple[tuple[Pos, ...], float, bool]:
        """Resolve Pacman/ghost contacts. Returns (ghosts, reward, dead)."""
        reward = 0.0
        new_ghosts = list(ghosts)
        for i, g in enumerate(ghosts):
            if g != pacman:
                continue
            if frightened > 0:
                reward += self.config.rewards.eat_ghost
                new_ghosts[i] = self.grid.ghost_starts[i]
            else:
                return ghosts, reward, True
        return tuple(new_ghosts), reward, False

    def _move_ghosts(
        self, pacman: Pos, ghosts: tuple[Pos, ...], frightened: int
    ) -> tuple[Pos, ...]:
        # One BFS from Pacman gives every ghost its distance to him.
        dist = self.grid.distances_from(pacman)
        moved = []
        for g in ghosts:
            options = [
                self.grid.move(g, a) for a in self.grid.legal_actions(g)
            ]
            if not options:
                moved.append(g)
                continue
            if self.rng.random() < self.config.ghost_random_prob:
                moved.append(self.rng.choice(options))
                continue
            far = len(dist) + 1
            if frightened > 0:  # flee: maximise distance to Pacman
                best = max(dist.get(o, far) for o in options)
            else:  # chase: minimise distance to Pacman
                best = min(dist.get(o, far) for o in options)
            ties = [o for o in options if dist.get(o, far) == best]
            moved.append(self.rng.choice(ties))
        return tuple(moved)

    def features(self, state: State) -> tuple[object, ...]:
        """Compact feature key for tabular agents on larger layout."""
        pac_dist = self.grid.distances_from(state.pacman)

        def _bucket(d: int) -> int:
            if d <= 1: return 0
            if d <= 3: return 1
            if d <= 6: return 2
            return 3

        def _sign(x: int) -> int:
            return (x > 0) - (x < 0)

        # All ghosts sorted by distance (closest first for stable ordering)
        ghost_features: list[object] = []
        for g in sorted(state.ghosts, key=lambda g: pac_dist.get(g, 999)):
            d = pac_dist.get(g, 999)
            ghost_features.append(_bucket(d))
            ghost_features.append((_sign(g[0] - state.pacman[0]),
                                   _sign(g[1] - state.pacman[1])))

        # Food features
        all_food = sorted(state.dots | state.pellets,
                          key=lambda f: pac_dist.get(f, 999))
        n_food = len(all_food)

        def _food_feature(f):
            d = pac_dist.get(f, 999)
            return (_bucket(d), (_sign(f[0] - state.pacman[0]),
                                 _sign(f[1] - state.pacman[1])))

        if n_food >= 1:
            d1, dir1 = _food_feature(all_food[0])
        else:
            d1, dir1 = 0, (0, 0)

        # Farthest food: helps track down the last isolated dots
        if n_food >= 1:
            df, dirf = _food_feature(all_food[-1])
        else:
            df, dirf = 0, (0, 0)

        # Remaining food count (bucketed)
        if n_food == 0:    food_count = 0
        elif n_food <= 5:  food_count = 1
        elif n_food <= 15: food_count = 2
        elif n_food <= 40: food_count = 3
        else:              food_count = 4

        legal = set(self.grid.legal_actions(state.pacman))
        walls = tuple(a not in legal for a in Action)

        return (*ghost_features,
                d1, dir1, df, dirf,
                food_count, state.frightened > 0, walls)
