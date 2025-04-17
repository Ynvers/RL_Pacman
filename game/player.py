#Creation of the class player
import pygame

class Player:
    def __init__(self, position: list, speed: float, size : int, score, history):
        self.position = position
        self.speed = speed
        self.size = size
        self.score = score
        self.history = history

    def move(self, direction):
        """
        A foction to update the postition of the pacman after a move
        """
        if direction == "UP":
            self.position[1] -= self.speed
        elif direction == "DOWN":
            self.position[1] += self.speed
        elif direction == "LEFT":
            self.position[0] -= self.speed
        elif direction == "RIGHT":
            self.position[0] += self.speed

    def draw(self, screen):
        """
        Draw pacman on the screen
        """
        pygame.draw.circle(screen, (255, 255, 0), (int(self.position[0]), int(self.position[1])), self.size)

    def check_collision(self, board_layout):
        """
        Check if a collision was be
        """
        cell_x = int(self.position[0] // 20)
        cell_y = int(self.position[1] // 20)
        if board_layout[cell_y][cell_x] == "#":
            return True
        elif board_layout[cell_y][cell_x] == ".":
            self.score += 10
            board_layout[cell_y][cell_x] = " "
        elif board_layout[cell_y][cell_x] == "o":
            self.score += 50
            board_layout[cell_y][cell_x] = " "
        elif board_layout[cell_y][cell_x] == "G":  
            self.score -= 100
            board_layout[cell_y][cell_x] = " "
        return False

    