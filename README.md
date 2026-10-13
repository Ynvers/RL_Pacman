# RL Pacman

Pacman rebuilt as a reinforcement learning environment, and a tabular
Q-learning agent that learns to play it.

The project started as a plain pygame Pacman with BFS ghosts. The game
logic now lives in a headless environment (`reset()` / `step(action)`),
so an agent can play thousands of games per second, and pygame is only
used to watch.

![screenshot](Visual_pacman.png)

## Quick start

```bash
uv sync --all-extras

uv run rl-pacman play                      # play yourself (arrows / ZQSD / WASD, R restart)
uv run rl-pacman train --layout small      # train, evaluate vs a random agent
uv run rl-pacman watch --layout small      # watch the trained agent
```

`train` writes `runs/latest/qtable.pkl`, `history.csv` and
`training.png` (reward and win-rate curves).

## The environment

| | |
|---|---|
| **State** | Pacman position, ghost positions, remaining dots and pellets, ghosts frightened or not |
| **Actions** | UP, DOWN, LEFT, RIGHT (walking into a wall = staying put) |
| **Rewards** | -1 per step, +10 dot, +50 pellet, +200 eaten ghost, +500 win, -500 death |
| **Ghosts** | Chase Pacman by BFS, take a random move 20% of the time, flee when frightened, move every other step |
| **End** | All dots eaten, caught by a ghost, or 300 steps |

Layouts: `tiny`, `small`, `classic` (in `src/rl_pacman/layouts.py`, easy
to add your own).

## Results

All runs use the default hyperparameters (`alpha=0.1`, `gamma=0.95`,
`epsilon_decay=0.9997`). Evaluation over 500 greedy episodes.

### Raw state key (no `--features`)

| Layout | Episodes | Win rate | Mean reward | States in Q-table |
|--------|----------|----------|-------------|-------------------|
| tiny   | 4 000    | ~99.6 %  | —           | ~400              |
| small  | 20 000   | ~91.8 %  | —           | ~80 000           |
| small  | 20 000 (random baseline) | 0.0 % | — | — |

On `small` the Q-table grows without bound and the reward plateaus
around −300 after 19 000 episodes: the state space is too large to
converge in a reasonable number of episodes.

### Feature key (`--features`)

Compact state: bucketed BFS distance + sign-direction to nearest ghost
and nearest food, plus a frightened boolean (≤ 2 592 distinct states).

| Layout  | Episodes | Win rate (train) | Win rate (greedy eval) | Mean reward (greedy) |
|---------|----------|------------------|------------------------|----------------------|
| classic | 50 000   | 0.0 %            | 0.0 %                  | −478                 |
| classic (random baseline) | — | — | 0.0 % | −476 |

The reward improves steadily during training (−457 → ~+250) and the
Q-table saturates at ~1 360 states, both signs that the agent is
learning. However the greedy evaluation matches random — see the next
section for why.

## Why tabular Q-learning stops at classic

**Raw state key — state space explosion.**
The state key includes the exact set of remaining dots as a `frozenset`.
On `small` (~20 dots) that is up to 2²⁰ ≈ 1 million dot configurations
alone. Most states are visited at most once; the Q-table never converges.

**Feature key — feature aliasing.**
Replacing the raw key with compact features (distance/direction to nearest
ghost and food) collapses the state space to ~2 600 entries. The Q-table
now converges and the training reward climbs. But many distinct board
positions map to the same feature vector: the agent cannot tell whether
it is in the top-left corridor or the bottom-right one.

In greedy evaluation this causes **loops**: the agent sees the same
feature state repeatedly and always picks the same action, cycling until
it walks into a ghost. During training, the 5 % residual exploration
accidentally breaks those loops — hence the positive training reward.
Remove exploration and the policy collapses to random performance.

**What would actually work on `classic`.**
The aliasing problem calls for a richer representation:

- Add wall bits (4 binary flags for the four neighbours) to the feature
  key — still tabular, ~40 k states, breaks most loops.
- Or switch to a neural network (Deep Q-Network / DQN): the network
  generalises across positions without hand-crafted features, and handles
  the full `classic` board.

## Project layout

```
src/rl_pacman/
  grid.py     maze parsing, movement, tunnels, BFS distances
  env.py      game rules and rewards (no pygame)
  agents.py   random baseline + Q-learning agent
  train.py    training / evaluation loops, CSV and plots
  render.py   pygame front-end
  cli.py      play / train / watch
tests/        environment rules
```

Checks: `uv run pytest`, `uv run mypy`, `uv run flake8 src tests`.
