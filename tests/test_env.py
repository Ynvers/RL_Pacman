from rl_pacman.env import Action, EnvConfig, PacmanEnv, Rewards
from rl_pacman.grid import Grid
from rl_pacman.layouts import LAYOUTS

R = Rewards()


def make(layout: str, **kw: object) -> PacmanEnv:
    # Ghosts effectively frozen unless a test asks otherwise.
    cfg = {"ghost_period": 10_000, "ghost_random_prob": 0.0, **kw}
    config = EnvConfig(layout=layout, **cfg)  # type: ignore[arg-type]
    return PacmanEnv(config, seed=0)


def test_all_layouts_parse() -> None:
    for layout in LAYOUTS.values():
        Grid.from_layout(layout)


def test_wall_bump_keeps_position() -> None:
    env = make("#####\n#P..#\n#####")
    env.reset()
    res = env.step(Action.UP)
    assert res.state.pacman == (1, 1)
    assert res.reward == R.step


def test_eating_a_dot() -> None:
    env = make("#####\n#P..#\n#####")
    env.reset()
    res = env.step(Action.RIGHT)
    assert res.state.pacman == (1, 2)
    assert (1, 2) not in res.state.dots
    assert res.reward == R.step + R.dot
    assert not res.done


def test_win_when_board_is_empty() -> None:
    env = make("####\n#P.#\n####")
    env.reset()
    res = env.step(Action.RIGHT)
    assert res.done and res.outcome == "win"
    assert res.reward == R.step + R.dot + R.win


def test_walking_into_a_ghost_kills() -> None:
    env = make("#####\n#PG.#\n#####")
    env.reset()
    res = env.step(Action.RIGHT)
    assert res.done and res.outcome == "death"


def test_ghost_chases_pacman() -> None:
    env = make("#######\n#P...G#\n#######", ghost_period=1)
    env.reset()
    res = env.step(Action.UP)  # wall bump, Pacman stays
    assert res.state.ghosts == ((1, 4),)


def test_tunnel_wraps_around() -> None:
    env = make("#####\nP . .\n#####")
    env.reset()
    res = env.step(Action.LEFT)
    assert res.state.pacman == (1, 4)
    assert res.reward == R.step + R.dot


def test_pellet_makes_ghosts_edible() -> None:
    env = make("######\n#PoG.#\n######")
    env.reset()
    res = env.step(Action.RIGHT)
    assert res.state.frightened > 0
    res = env.step(Action.RIGHT)
    assert not res.done
    assert res.reward == R.step + R.eat_ghost
    assert res.state.ghosts == ((1, 3),)  # respawned at its start


def test_timeout() -> None:
    env = make("#####\n#P..#\n#####", max_steps=3)
    env.reset()
    for _ in range(3):
        res = env.step(Action.UP)
    assert res.done and res.outcome == "timeout"


def test_classic_tunnel_row() -> None:
    env = make(LAYOUTS["classic"])
    env.reset()
    start = env.state.pacman
    assert start == (9, 0)
    res = env.step(Action.LEFT)
    assert res.state.pacman == (9, env.grid.width - 1)
