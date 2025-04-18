import pygame

class Board:
    def __init__(self, layout, cell_size, wall_color, point_color):
        self.layout = layout
        self.cell_size = cell_size
        self.wall_color = wall_color
        self.point_color = point_color

    def draw(self, screen):
        """
        Draw the board on the screen.
        """
        for row_index, row in enumerate(self.layout):
            for col_index, cell in enumerate(row):
                x = col_index * self.cell_size
                y = row_index * self.cell_size
                if cell == "#":
                    pygame.draw.rect(screen, self.wall_color, ((x, y), (self.cell_size, self.cell_size)))
                elif cell == ".":
                    pygame.draw.circle(screen, self.point_color, (x + self.cell_size // 2, y + self.cell_size // 2), 3)
                elif cell == "o":
                    pygame.draw.circle(screen, self.point_color, (x + self.cell_size // 2, y + self.cell_size // 2), 6)
                elif cell == " ":
                    pygame.draw.rect(screen, (0, 0, 0), ((x, y), (self.cell_size, self.cell_size)), 1)
                elif cell == "P":
                    pygame.draw.rect(screen, (255, 255, 0), ((x, y), (self.cell_size, self.cell_size)))
                elif cell == "G":   
                    pygame.draw.rect(screen, (255, 0, 0), ((x, y), (self.cell_size, self.cell_size)))

    def get_player_start_position(self):
        """
        Get the player's start position on the board
        """
        for row_index, row in enumerate(self.layout):
            for col_index, cell in enumerate(row):
                if cell == "P":
                    self.layout[row_index][col_index] = " "
                    #return [col_index * self.cell_size, row_index * self.cell_size]    
                    return [
                        col_index * self.cell_size + self.cell_size // 2 - self.cell_size // 2,
                        row_index * self.cell_size + self.cell_size // 2 - self.cell_size // 2
                    ]
        return None
            
#Layout of the board
board_layout = [
    list("############################"),
    list("#............##............#"),
    list("#.####.#####.##.#####.####.#"),
    list("#o####.#####.##.#####.####o#"),
    list("#.####.#####.##.#####.####.#"),
    list("#..........................#"),
    list("#####.##.##########.##.#####"),
    list("    #.##.##########.##.#    "),
    list("#####.##.###    ###.##.#####"),
    list("P.....##.###    ###.##......"),
    list("#####.##.##########.##.#####"),
    list("    #.##.##########.##.#    "),
    list("#####.##.##########.##.#####"),
    list("#............##............#"),
    list("#.####.#####.##.#####.####.#"),
    list("#o####.#####.##.#####.####o#"),
    list("#...##................##...#"),
    list("###.##.###########.##.##.###"),
    list("###.##.###########.##.##.###"),
    list("#..........................#"),
    list("############################")
]