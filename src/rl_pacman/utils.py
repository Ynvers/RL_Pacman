from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rl_pacman.env import State
    from rl_pacman.grid import Grid


def encode_state(features: Iterable[object]) -> list[float]:
    encoded_features: list[float] = []
    for feature in features:
        if isinstance(feature, (tuple, list)):
            encoded_features.extend(encode_state(feature))
        else:
            encoded_features.append(float(feature))
    return encoded_features


def encode_state_cnn(state: State, grid: Grid) -> list[list[list[float]]]:
    """Encode game state as a (6, H, W) grid for CNN input.

    Channels:
        0 — walls
        1 — Pacman position
        2 — dangerous ghosts (frightened == 0)
        3 — frightened ghosts (frightened > 0)
        4 — remaining dots
        5 — remaining pellets
    """
    H, W = grid.height, grid.width
    channels: list[list[list[float]]] = [
        [[0.0] * W for _ in range(H)] for _ in range(6)
    ]

    for r, c in grid.walls:
        channels[0][r][c] = 1.0

    r, c = state.pacman
    channels[1][r][c] = 1.0

    ghost_ch = 3 if state.frightened > 0 else 2
    for r, c in state.ghosts:
        channels[ghost_ch][r][c] = 1.0

    for r, c in state.dots:
        channels[4][r][c] = 1.0

    for r, c in state.pellets:
        channels[5][r][c] = 1.0

    return channels


if __name__ == "__main__":
    features = (2, (-1, 0), 1, (1, 1), False)
    print(encode_state(features))