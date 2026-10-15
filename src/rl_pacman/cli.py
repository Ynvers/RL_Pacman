"""Command line: `rl-pacman play | train | train-dql | watch`."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl_pacman.agents import DQLAgentCNNPDD, QLearningAgent, RandomAgent
from rl_pacman.env import EnvConfig, PacmanEnv, State
from rl_pacman.layouts import LAYOUTS


def _env(args: argparse.Namespace) -> PacmanEnv:
    return PacmanEnv(EnvConfig(layout=args.layout), seed=args.seed)


def cmd_play(args: argparse.Namespace) -> None:
    from rl_pacman import render

    render.run(_env(args))


def cmd_train(args: argparse.Namespace) -> None:
    from rl_pacman.train import evaluate, plot_history, save_history, train

    env = _env(args)
    agent = QLearningAgent(
        env.n_actions,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )
    key_fn = (lambda s: env.features(s)) if args.features else None
    history = train(env, agent, args.episodes, log_every=args.log_every,
                    key_fn=key_fn)

    out = Path(args.out)
    agent.save(out / "qtable.pkl")
    save_history(history, out / "history.csv")
    try:
        plot_history(history, out / "training.png")
    except ImportError:
        print("(install the `plot` extra to get training.png)")

    trained = evaluate(env, agent, args.eval_episodes)
    baseline = evaluate(env, RandomAgent(env.n_actions, seed=args.seed),
                        args.eval_episodes)
    print(f"\n{'':10}{'win':>8}{'death':>8}{'reward':>10}{'steps':>8}")
    for name, m in (("random", baseline), ("q-learning", trained)):
        print(f"{name:10}{m['win_rate']:8.1%}{m['death_rate']:8.1%}"
              f"{m['mean_reward']:10.1f}{m['mean_steps']:8.1f}")
    print(f"\nsaved to {out}/")


def cmd_train_dql(args: argparse.Namespace) -> None:
    import numpy as np
    from rl_pacman.train import evaluate, plot_history, save_history, train_dql
    from rl_pacman.utils import encode_state_cnn

    env = _env(args)
    agent = DQLAgentCNNPDD(
        env,
        capacity=args.capacity,
        n_actions=env.n_actions,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )
    # store compact numpy arrays in the buffer, not nested Python lists
    encode_fn = lambda s: np.array(encode_state_cnn(s, env.grid), dtype=np.float32)
    history = train_dql(
        env, agent, args.episodes,
        target_update=args.target_update,
        learn_every=args.learn_every,
        log_every=args.log_every,
        encode_fn=encode_fn,
    )

    import torch
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(agent.policy_net.state_dict(), out / "policy_net.pt")
    save_history(history, out / "history_pdd.csv")
    try:
        plot_history(history, out / "training_pdd.png")
    except ImportError:
        print("(install the `plot` extra to get training_pdd.png)")

    trained = evaluate(env, agent, args.eval_episodes, key_fn=encode_fn)
    baseline = evaluate(env, RandomAgent(env.n_actions, seed=args.seed), args.eval_episodes)
    print(f"\n{'':10}{'win':>8}{'death':>8}{'reward':>10}{'steps':>8}")
    for name, m in (("random", baseline), ("pdd-dqn", trained)):
        print(f"{name:10}{m['win_rate']:8.1%}{m['death_rate']:8.1%}"
              f"{m['mean_reward']:10.1f}{m['mean_steps']:8.1f}")
    print(f"\nsaved to {out}/")


def cmd_watch(args: argparse.Namespace) -> None:
    from rl_pacman import render

    env = _env(args)
    agent = QLearningAgent(env.n_actions, seed=args.seed)
    agent.load(Path(args.qtable))

    def policy(state: State) -> int:
        return agent.act(state.key(), greedy=True)

    render.run(env, policy)


def main() -> None:
    parser = argparse.ArgumentParser(prog="rl-pacman")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, default_layout: str) -> None:
        p.add_argument("--layout", default=default_layout,
                       choices=sorted(LAYOUTS))
        p.add_argument("--seed", type=int, default=None)

    p = sub.add_parser("play", help="play with the keyboard")
    common(p, "classic")
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("train", help="train a Q-learning agent")
    common(p, "small")
    p.add_argument("--episodes", type=int, default=20_000)
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--epsilon-decay", type=float, default=0.9997)
    p.add_argument("--eval-episodes", type=int, default=500)
    p.add_argument("--log-every", type=int, default=1000)
    p.add_argument("--out", default="runs/latest")
    p.add_argument("--features", action="store_true",
                   help="use compact feature key instead of raw state")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("train-dql", help="train a Prioritized Double Dueling DQN agent (CNN)")
    common(p, "classic")
    p.add_argument("--episodes", type=int, default=5_000)
    p.add_argument("--capacity", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--gamma", type=float, default=0.95)
    p.add_argument("--epsilon-decay", type=float, default=0.9998)
    p.add_argument("--target-update", type=int, default=500)
    p.add_argument("--learn-every", type=int, default=4)
    p.add_argument("--eval-episodes", type=int, default=100)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--out", default="runs/latest")
    p.set_defaults(func=cmd_train_dql)

    p = sub.add_parser("watch", help="watch a trained agent play")
    common(p, "small")
    p.add_argument("--qtable", default="runs/latest/qtable.pkl")
    p.set_defaults(func=cmd_watch)

    args = parser.parse_args()
    try:
        args.func(args)
    except NotImplementedError as e:
        raise SystemExit(
            f"{e}: see the TODOs in src/rl_pacman/agents.py"
        ) from None


if __name__ == "__main__":
    main()
