"""Agents: a random baseline and a tabular Q-learning agent."""

from __future__ import annotations

import pickle
import random
from collections.abc import Hashable
from pathlib import Path
from torch import nn
import torch
from collections import deque


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
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
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
        """Choose an action."""
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)

        q_vals =  self.values(state_key)
        best = max(q_vals)
        ties = [a for a, v in enumerate(q_vals) if v == best]
        return self.rng.choice(ties)

    def update(
        self,
        state_key: Hashable,
        action: int,
        reward: float,
        next_state_key: Hashable,
        done: bool,
    ) -> None:
        """Learn from one transition (s, a, r, s', done).
        """
        q_current = self.values(state_key)[action]

        if done:
            target = reward
        else:
            q_next = max(self.values(next_state_key))
            target = reward + self.gamma * q_next

        self.values(state_key)[action] = q_current + self.alpha * (target - q_current)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self.q, f)

    def load(self, path: Path) -> None:
        with path.open("rb") as f:
            self.q = pickle.load(f)


class DQNetwork(nn.Module):

    def __init__(
        self,
        n_actions: int,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.n_actions = n_actions
        self.rng = random.Random(seed)
        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, n_actions)

    def forward(self, x):
        result = self.fc1(x)
        result = nn.functional.relu(result)
        return self.fc2(result)


class ReplayBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQLAgent:
    def __init__(
        self,
        capacity,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.999,
        seed: int | None = None,
    ) -> None:
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        self.policy_net = DQNetwork(n_actions)
        self.target_net = DQNetwork(n_actions)
        self.buffer = ReplayBuffer(capacity)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def act(self, state: list[float], greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)

        x = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            q_values = self.policy_net(x)

        return q_values.argmax().item()

    def learn(self, batch_size: int = 64) -> None:
        if len(self.buffer) < batch_size:
            return

        transitions = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*transitions)
        states = torch.tensor(states, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        predicted = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1).values
        targets = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.MSELoss()(predicted, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())
