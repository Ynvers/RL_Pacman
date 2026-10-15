"""Agents: a random baseline and a tabular Q-learning agent."""

from __future__ import annotations

import pickle
import random
from collections.abc import Hashable
from pathlib import Path
from torch import nn
import torch
import numpy as np
from collections import deque

from rl_pacman.env import PacmanEnv
from rl_pacman.utils import encode_state


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
        n_inputs: int,
        n_actions: int,
    ) -> None:
        super().__init__()
        self.fc1 = nn.Linear(n_inputs, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, n_actions)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.relu(self.fc2(x))
        return self.fc3(x)


class DQNetworkCNN(nn.Module):
    """Convolutional network: takes the full (C, H, W) game grid as input.

    Two conv+pool blocks halve the spatial dims twice:
    (C, H, W) → (16, H/2, W/2) → (32, H/4, W/4) → flatten → FC.
    """

    def __init__(self, n_channels: int, height: int, width: int, n_actions: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        h_out = height // 4
        w_out = width // 4
        self.fc = nn.Sequential(
            nn.Linear(32 * h_out * w_out, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)


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
        env: PacmanEnv,
        capacity,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.999,
        seed: int | None = None,
    ) -> None:
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        sample = encode_state(env.features(env.reset()))
        n_inputs = len(sample)
        self.policy_net = DQNetwork(n_inputs, n_actions)
        self.target_net = DQNetwork(n_inputs, n_actions)
        self.buffer = ReplayBuffer(capacity)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
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
            # Double DQN: policy_net selects the action, target_net evaluates it
            best_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
        targets = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.SmoothL1Loss()(predicted, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())


class DQLAgentCNN:
    """DQL agent using a convolutional network on the full game grid."""

    def __init__(
        self,
        env: PacmanEnv,
        capacity: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.999,
        seed: int | None = None,
    ) -> None:
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        n_channels = 6
        H, W = env.grid.height, env.grid.width
        self.policy_net = DQNetworkCNN(n_channels, H, W, n_actions)
        self.target_net = DQNetworkCNN(n_channels, H, W, n_actions)
        self.buffer = ReplayBuffer(capacity)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

    def act(self, state: list, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)
        # unsqueeze(0) adds the batch dimension required by Conv2d
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(x)
        return q_values.squeeze(0).argmax().item()

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
            best_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
        targets = rewards + self.gamma * next_q * (1 - dones)

        loss = nn.SmoothL1Loss()(predicted, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())


# ---------------------------------------------------------------------------
# Prioritized Double Dueling DQN (PDD)
# ---------------------------------------------------------------------------

class SumTree:
    """Binary tree where each leaf stores a priority; internal nodes store sums.

    Supports O(log n) add/update and O(log n) stratified sampling.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data: list = [None] * capacity
        self.write = 0
        self.n_entries = 0

    def _propagate(self, idx: int, delta: float) -> None:
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data: object) -> None:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        self._propagate(idx, priority - self.tree[idx])
        self.tree[idx] = priority

    def get(self, s: float) -> tuple[int, float, object]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]


class PrioritizedReplayBuffer:
    """Replay buffer that samples transitions proportional to TD-error priority."""

    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_steps: int = 2_000_000,
    ) -> None:
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = (1.0 - beta) / beta_steps
        self.epsilon = 1e-6
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done) -> None:
        # New transitions get max priority so they are sampled at least once.
        self.tree.add(self.max_priority, (state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        indices, priorities, batch = [], [], []
        segment = self.tree.total / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, priority, data = self.tree.get(s)
            indices.append(idx)
            priorities.append(priority)
            batch.append(data)

        probs = np.maximum(np.array(priorities) / self.tree.total, 1e-10)
        weights = (self.tree.n_entries * probs) ** (-self.beta)
        weights /= weights.max()

        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
            indices,
            torch.tensor(weights, dtype=torch.float32),
        )

    def update_priorities(self, indices: list[int], td_errors: np.ndarray) -> None:
        for idx, err in zip(indices, td_errors):
            priority = (float(abs(err)) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return self.tree.n_entries


class DQNetworkCNNDueling(nn.Module):
    """CNN backbone + dueling streams: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)."""

    def __init__(self, n_channels: int, height: int, width: int, n_actions: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        flat = 64 * (height // 4) * (width // 4)
        self.value = nn.Sequential(nn.Linear(flat, 256), nn.ReLU(), nn.Linear(256, 1))
        self.advantage = nn.Sequential(nn.Linear(flat, 256), nn.ReLU(), nn.Linear(256, n_actions))

    def forward(self, x):
        x = self.conv(x).flatten(1)
        v = self.value(x)
        a = self.advantage(x)
        return v + a - a.mean(dim=1, keepdim=True)


class DQLAgentCNNPDD:
    """Prioritized Double Dueling DQN agent with CNN.

    Combines:
    - Double DQN (policy net selects action, target net evaluates it)
    - Dueling architecture (separate V and A streams)
    - Prioritized Experience Replay (high-error transitions sampled more often)
    """

    def __init__(
        self,
        env: PacmanEnv,
        capacity: int = 100_000,
        n_actions: int = 5,
        lr: float = 2.5e-4,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.10,
        epsilon_decay: float = 0.9998,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_steps: int = 2_000_000,
        seed: int | None = None,
    ) -> None:
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        H, W = env.grid.height, env.grid.width
        self.policy_net = DQNetworkCNNDueling(6, H, W, n_actions)
        self.target_net = DQNetworkCNNDueling(6, H, W, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.buffer = PrioritizedReplayBuffer(capacity, alpha=alpha, beta=beta, beta_steps=beta_steps)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    def act(self, state: list, greedy: bool = False) -> int:
        if not greedy and self.rng.random() < self.epsilon:
            return self.rng.randrange(self.n_actions)
        x = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(x)
        return q_values.squeeze(0).argmax().item()

    def learn(self, batch_size: int = 64) -> None:
        if len(self.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones, indices, weights = self.buffer.sample(batch_size)

        with torch.no_grad():
            best_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, best_actions).squeeze(1)
            targets = rewards + self.gamma * next_q * (1 - dones)

        predicted = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        td_errors = (targets - predicted).detach().abs().cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)

        # IS weights compensate for PER sampling bias
        loss = (weights * nn.functional.huber_loss(predicted, targets, reduction='none')).mean()
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())
