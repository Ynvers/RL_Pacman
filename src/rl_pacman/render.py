"""pygame front-end: play yourself, or watch an agent play.

The renderer only *reads* the environment; all game rules live in env.py.
"""

from __future__ import annotations

from collections.abc import Callable

import pygame

from rl_pacman.env import Action, PacmanEnv, State

CELL = 24
HUD = 40
STEPS_PER_SECOND = 8

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
WALL = (33, 33, 222)
DOT = (255, 184, 151)
PACMAN = (255, 255, 0)
FRIGHTENED = (60, 60, 255)
GHOST_COLORS = [
    (255, 0, 0), (255, 184, 255), (0, 255, 255), (255, 184, 82),
]

KEYS: dict[int, Action] = {
    pygame.K_UP: Action.UP,
    pygame.K_z: Action.UP,
    pygame.K_w: Action.UP,
    pygame.K_DOWN: Action.DOWN,
    pygame.K_s: Action.DOWN,
    pygame.K_LEFT: Action.LEFT,
    pygame.K_q: Action.LEFT,
    pygame.K_a: Action.LEFT,
    pygame.K_RIGHT: Action.RIGHT,
    pygame.K_d: Action.RIGHT,
}

Policy = Callable[[State], int]


def _draw(screen: pygame.Surface, env: PacmanEnv, font: pygame.font.Font,
          message: str) -> None:
    g, s = env.grid, env.state
    screen.fill(BLACK)
    half = CELL // 2

    def center(pos: tuple[int, int]) -> tuple[int, int]:
        return pos[1] * CELL + half, pos[0] * CELL + half + HUD

    for r, c in g.walls:
        pygame.draw.rect(screen, WALL, (c * CELL, r * CELL + HUD, CELL, CELL))
    for d in s.dots:
        pygame.draw.circle(screen, DOT, center(d), 3)
    for p in s.pellets:
        pygame.draw.circle(screen, DOT, center(p), 7)
    pygame.draw.circle(screen, PACMAN, center(s.pacman), half - 2)
    for i, gh in enumerate(s.ghosts):
        color = FRIGHTENED if s.frightened else GHOST_COLORS[i % 4]
        x, y = center(gh)
        pygame.draw.rect(screen, color, (x - half + 2, y - half + 2,
                                         CELL - 4, CELL - 4),
                         border_radius=8)

    hud = font.render(
        f"Score {env.score:.0f}   Steps {env.steps}", True, WHITE
    )
    screen.blit(hud, (8, 8))
    if message:
        text = font.render(message, True, WHITE)
        rect = text.get_rect(center=screen.get_rect().center)
        pygame.draw.rect(screen, BLACK, rect.inflate(20, 12))
        screen.blit(text, rect)
    pygame.display.flip()


def run(env: PacmanEnv, policy: Policy | None = None,
        max_frames: int | None = None) -> None:
    """Open a window. `policy=None` means keyboard control.

    R restarts, Esc quits. `max_frames` is only for headless smoke tests.
    """
    pygame.init()
    screen = pygame.display.set_mode(
        (env.grid.width * CELL, env.grid.height * CELL + HUD)
    )
    pygame.display.set_caption("RL Pacman")
    font = pygame.font.Font(None, 28)
    clock = pygame.time.Clock()

    env.reset()
    direction = Action.LEFT
    message = ""
    frames = 0
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    env.reset()
                    message = ""
                elif event.key in KEYS:
                    direction = KEYS[event.key]

        if not env.done:
            action = direction if policy is None else policy(env.state)
            result = env.step(action)
            if result.done:
                message = {
                    "win": "YOU WIN!",
                    "death": "GAME OVER",
                    "timeout": "TIME UP",
                }[result.outcome] + "   (R to restart)"

        _draw(screen, env, font, message)
        clock.tick(STEPS_PER_SECOND)
        frames += 1
        if max_frames is not None and frames >= max_frames:
            running = False

    pygame.quit()
