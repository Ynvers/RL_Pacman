#Creation of the class player
import pygame
import time

class Player:
    def __init__(self, position: list, speed: float, size : int, history : list):
        self.position = [position[0], position[1]]
        self.position[0] = round(position[0] / size) * size
        self.position[1] = round(position[1] / size) * size
        self.speed = speed
        self.size = size
        self.score = 0
        self.history = history
        self.last_move_time = time.time()
        self.game_won = False

    def draw(self, screen):
        """
        Draw pacman on the screen
        """
        #pygame.draw.circle(screen, (255, 255, 0), (int(self.position[0]), int(self.position[1])), self.size)
        pygame.draw.rect(screen, (255, 255, 0), ((self.position[0], self.position[1]), (self.size, self.size)))

    def move(self, direction, board_layout):
        """
        Met à jour la position de Pacman après un mouvement.
        """
        current_time = time.time()
        if current_time - self.last_move_time < 1 / self.speed:
            return  # Trop tôt pour un nouveau mouvement

        self.last_move_time = current_time

        # Convertir la position actuelle en coordonnées de grille
        cell_x = int(self.position[0] // self.size)
        cell_y = int(self.position[1] // self.size)

        # Calculer la nouvelle position en fonction de la direction
        if direction == "UP":
            new_cell_x, new_cell_y = cell_x, cell_y - 1
        elif direction == "DOWN":
            new_cell_x, new_cell_y = cell_x, cell_y + 1
        elif direction == "LEFT":
            if cell_x == 0 and cell_y == 9:
                print("Teleporting to the right side")
                new_cell_x, new_cell_y = len(board_layout[0]) - 1, cell_y
            else:
                new_cell_x, new_cell_y = cell_x - 1, cell_y
        elif direction == "RIGHT":
            if cell_x == len(board_layout[0]) - 1 and cell_y == 9:
                print("Teleporting to the left side")
                new_cell_x, new_cell_y = 0, cell_y
            else:
                new_cell_x, new_cell_y = cell_x + 1, cell_y
        else:
            return

        # Vérifier si la nouvelle position est un mur
        if board_layout[new_cell_y][new_cell_x] != "#":
            self.position[0] = new_cell_x * self.size
            self.position[1] = new_cell_y * self.size
            print(f"New position (pixels): ({self.position[0]}, {self.position[1]})")

    def check_collision(self, board_layout):
        """
        Check if a collision was be
        """
        cell_x = int(self.position[0] // self.size)
        cell_y = int(self.position[1] // self.size)
        
        if board_layout[cell_y][cell_x] == "#":
            print("Collision with wall")
            return True
        elif board_layout[cell_y][cell_x] == ".":
            print("Collision with point")
            self.score += 1
            board_layout[cell_y][cell_x] = " "  # Effacer le point
            print(f"Score updated: {self.score}")
        elif board_layout[cell_y][cell_x] == "o":
            print("Collision with big point")
            self.score += 5
            board_layout[cell_y][cell_x] = " "  # Effacer le gros point
            print(f"Score updated: {self.score}")
        elif board_layout[cell_y][cell_x] == "G":
            print("Collision with ghost")
            self.score -= 100
            board_layout[cell_y][cell_x] = " "
            print(f"Score updated: {self.score}")
        return False

