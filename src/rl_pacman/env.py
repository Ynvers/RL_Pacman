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
