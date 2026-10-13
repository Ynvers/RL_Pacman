"""Maze layouts.

Legend:
    #  wall
    .  dot
    o  power pellet (ghosts become frightened and can be eaten)
    P  Pacman start
    G  ghost start
    (space) empty floor

A row whose edge cell is not a wall acts as a tunnel: leaving the grid on
one side brings you back on the other side.
"""

CLASSIC = """\
############################
#G...........##...........G#
#.####.#####.##.#####.####.#
#o####.#####.##.#####.####o#
#.####.#####.##.#####.####.#
#..........................#
#####.##.#.##..##.#.##.#####
    #.##.#.##..##.#.##.#
#####.##.#.##..##.#.##.#####
P.....##.#.##..##.#.##......
#####.##.#.##..##.#.##.#####
    #.##.#..........##.#
#####.##.#.########.##.#####
#............##............#
#.####.#####.##.#####.####.#
#o####.#####.##.#####.####o#
#...##.......G........##...#
###.##.###########.##.##.###
###.##.###########.##.##.###
#..........................#
############################"""

# Small enough for a tabular agent: few dots, one ghost.
SMALL = """\
#########
#P.....o#
#.##.##.#
#...#...#
#.#...#.#
#o....G.#
#########"""

TINY = """\
#######
#P...o#
#.#.#.#
#...G.#
#######"""

LAYOUTS: dict[str, str] = {
    "classic": CLASSIC,
    "small": SMALL,
    "tiny": TINY,
}
