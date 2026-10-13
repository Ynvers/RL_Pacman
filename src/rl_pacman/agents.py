"""Agents: a random baseline and a tabular Q-learning agent."""

from __future__ import annotations

import pickle
import random
from collections.abc import Hashable
from pathlib import Path


class RandomAgent:
    """Baseline: picks a uniformly random action every step."""

    def __init__(self, n_actions: int, seed: int | None = None) -> None:
        self.n_actions = n_actions
        self.rng = random.Random(seed)

    def act(self, state_key: Hashable, greedy: bool = False) -> int:
        return self.rng.randrange(self.n_actions)


class QLearningAgent:
    """Tabular Q-learning with an epsilon-greedy policy.

    The Q-table maps a state key to a list of `n_actions` values:
    `self.q[state_key][action]` is the estimate of the total discounted
    reward obtained by taking `action` in that state and acting well after.
    """

    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.999,
        seed: int | None = None,
    ) -> None:
        self.n_actions = n_actions
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        self.q: dict[Hashable, list[float]] = {}

    def values(self, state_key: Hashable) -> list[float]:
        """Q-values of a state (all zeros if never seen)."""
        if state_key not in self.q:
            self.q[state_key] = [0.0] * self.n_actions
        return self.q[state_key]

    def act(self, state_key: Hashable, greedy: bool = False) -> int:
        """Choose an action.

        TODO(Ynvers): epsilon-greedy.
          - if not greedy, with probability self.epsilon: random action
            (use self.rng so runs are reproducible)
          - otherwise: the action with the highest Q-value in this state.
            Several actions can share the max (at the start everything is
            0.0): break ties randomly, otherwise the agent always goes UP.
        """
        raise NotImplementedError("act() is yours to write")

    def update(
        self,
        state_key: Hashable,
        action: int,
        reward: float,
        next_state_key: Hashable,
        done: bool,
    ) -> None:
        """Learn from one transition (s, a, r, s', done).

        TODO(Ynvers): the Q-learning update.
          - target: what this step was "worth" = immediate reward
            + discounted value of the best action in the next state.
            If the episode ended (done), there is no future: just reward.
          - move Q(s, a) a fraction self.alpha of the way toward the target.
        """
        raise NotImplementedError("update() is yours to write")

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.q, f)

    def load(self, path: Path) -> None:
        with path.open("rb") as f:
            self.q = pickle.load(f)
