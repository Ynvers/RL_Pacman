"""Training and evaluation loops."""

from __future__ import annotations

import csv
from collections.abc import Callable, Hashable
from rl_pacman.env import State
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from collections.abc import Callable

from rl_pacman.agents import DQLAgent, DQLAgentCNN, DQLAgentCNNPDD, QLearningAgent
from rl_pacman.env import PacmanEnv, State
from rl_pacman.utils import encode_state


class Agent(Protocol):
    def act(self, state_key: Hashable, greedy: bool = False) -> int: ...


@dataclass(frozen=True)
class EpisodeStats:
    episode: int
    total_reward: float
    steps: int
    outcome: str
    dots_left: int
    epsilon: float


def run_episode(
    env: PacmanEnv,
    agent: Agent,
    episode: int = 0,
    learn: bool = False,
    greedy: bool = False,
    key_fn: Callable[[State], Hashable] | None = None,
) -> EpisodeStats:
    if key_fn is None:
        key_fn = lambda s: s.key()
    state = env.reset()
    total = 0.0
    while True:
        key = key_fn(state)
        action = agent.act(key, greedy=greedy)
        result = env.step(action)
        if learn and isinstance(agent, QLearningAgent):
            agent.update(
                key, action, result.reward, key_fn(result.state), result.done
            )
        total += result.reward
        state = result.state
        if result.done:
            break
    eps = agent.epsilon if isinstance(agent, QLearningAgent) else 1.0
    return EpisodeStats(
        episode=episode,
        total_reward=total,
        steps=env.steps,
        outcome=result.outcome,
        dots_left=len(state.dots) + len(state.pellets),
        epsilon=eps,
    )


def train(
    env: PacmanEnv,
    agent: QLearningAgent,
    episodes: int,
    log_every: int = 500,
    key_fn: Callable[[State], Hashable] | None = None,
) -> list[EpisodeStats]:
    history: list[EpisodeStats] = []
    for ep in range(1, episodes + 1):
        stats = run_episode(env, agent, episode=ep, learn=True, key_fn=key_fn)
        agent.decay_epsilon()
        history.append(stats)
        if log_every and ep % log_every == 0:
            window = history[-log_every:]
            mean = sum(s.total_reward for s in window) / len(window)
            wins = sum(s.outcome == "win" for s in window) / len(window)
            print(
                f"episode {ep:>6}  mean reward {mean:8.1f}  "
                f"win rate {wins:5.1%}  epsilon {agent.epsilon:.3f}  "
                f"states {len(agent.q)}"
            )
    return history

def train_dql(
    env: PacmanEnv,
    agent: DQLAgent | DQLAgentCNN | DQLAgentCNNPDD,
    episodes: int,
    target_update: int = 100,
    learn_every: int = 4,
    log_every: int = 500,
    encode_fn: Callable[[State], list] | None = None,
) -> list[EpisodeStats]:
    """Train a DQN agent with replay and a periodically synced target network.

    encode_fn converts a State to the encoded representation the agent expects.
    Defaults to the hand-crafted feature vector; pass encode_state_cnn for CNN.
    """
    if encode_fn is None:
        encode_fn = lambda s: encode_state(env.features(s))
    if target_update <= 0:
        raise ValueError("target_update must be greater than zero")
    if episodes < 0:
        raise ValueError("episodes must not be negative")

    history: list[EpisodeStats] = []
    step_count = 0

    for ep in range(1, episodes + 1):
        state = env.reset()
        encoded = encode_fn(state)
        total_reward = 0.0

        while True:
            action = agent.act(encoded)
            result = env.step(action)

            next_encoded = encode_fn(result.state)
            agent.buffer.push(
                encoded, action, result.reward, next_encoded, result.done
            )
            step_count += 1
            if step_count % learn_every == 0:
                agent.learn()
            if step_count % target_update == 0:
                agent.update_target()

            total_reward += result.reward
            encoded = next_encoded
            state = result.state
            if result.done:
                break

        stats = EpisodeStats(
            episode=ep,
            total_reward=total_reward,
            steps=env.steps,
            outcome=result.outcome,
            dots_left=len(state.dots) + len(state.pellets),
            epsilon=agent.epsilon,
        )
        history.append(stats)
        agent.decay_epsilon()

        if log_every and ep % log_every == 0:
            window = history[-log_every:]
            mean = sum(s.total_reward for s in window) / len(window)
            wins = sum(s.outcome == "win" for s in window) / len(window)
            mean_dots = sum(s.dots_left for s in window) / len(window)
            print(
                f"episode {ep:>6}  mean reward {mean:8.1f}  "
                f"win rate {wins:5.1%}  epsilon {agent.epsilon:.3f}  "
                f"dots_left {mean_dots:5.1f}"
            )

    return history

def evaluate(env: PacmanEnv, agent: Agent, episodes: int) -> dict[str, float]:
    """Play `episodes` greedy games without learning."""
    runs = [
        run_episode(env, agent, episode=i, greedy=True)
        for i in range(episodes)
    ]
    n = len(runs)
    return {
        "win_rate": sum(r.outcome == "win" for r in runs) / n,
        "death_rate": sum(r.outcome == "death" for r in runs) / n,
        "mean_reward": sum(r.total_reward for r in runs) / n,
        "mean_steps": sum(r.steps for r in runs) / n,
    }


def save_history(history: list[EpisodeStats], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["episode", "total_reward", "steps", "outcome", "dots_left",
             "epsilon"]
        )
        for s in history:
            writer.writerow(
                [s.episode, s.total_reward, s.steps, s.outcome, s.dots_left,
                 round(s.epsilon, 4)]
            )


def plot_history(
    history: list[EpisodeStats], path: Path, window: int = 200
) -> None:
    """Reward and win-rate curves (moving average). Needs the `plot` extra."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def moving(values: list[float]) -> list[float]:
        out, acc = [], 0.0
        for i, v in enumerate(values):
            acc += v
            if i >= window:
                acc -= values[i - window]
            out.append(acc / min(i + 1, window))
        return out

    episodes = [s.episode for s in history]
    rewards = moving([s.total_reward for s in history])
    wins = moving([1.0 if s.outcome == "win" else 0.0 for s in history])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax1.plot(episodes, rewards)
    ax1.set_ylabel(f"reward (avg {window})")
    ax2.plot(episodes, wins)
    ax2.set_ylabel(f"win rate (avg {window})")
    ax2.set_xlabel("episode")
    ax2.set_ylim(0, 1)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
