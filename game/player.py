#Creation of the class player
import pygame

class Player:
    def __init__(self, position: list, speed: float, size : int, score : int, history : list):
        self.position = [position[0], position[1]]
        self.position[0] = round(position[0] / size) * size
        self.position[1] = round(position[1] / size) * size
        self.speed = speed
        self.size = size
        self.score = score
        self.history = history

    def draw(self, screen):
        """
        Draw pacman on the screen
        """
        #pygame.draw.circle(screen, (255, 255, 0), (int(self.position[0]), int(self.position[1])), self.size)
        pygame.draw.rect(screen, (255, 255, 0), ((self.position[0], self.position[1]), (self.size, self.size)))

    def move(self, direction, board_layout):
        """
        A foction to update the postition of the pacman after a move
        """
        cell_x = int(self.position[0] // self.size)
        cell_y = int(self.position[1] // self.size)
        
        if direction == "UP":
            new_cell_x, new_cell_y = cell_x, cell_y - 1
        elif direction == "DOWN":
            new_cell_x, new_cell_y = cell_x, cell_y + 1
        elif direction == "LEFT":
            new_cell_x, new_cell_y = cell_x - 1, cell_y
        elif direction == "RIGHT":
            new_cell_x, new_cell_y = cell_x + 1, cell_y
        else:
            return
        #
        self.position[0] = new_cell_x * self.size
        self.position[1] = new_cell_y * self.size

    def check_collision(self, board_layout):
        """
        Check if a collision was be
        """
        cell_x = int(self.position[0] // 20)
        cell_y = int(self.position[1] // 20)
        if board_layout[cell_y][cell_x] == "#":
            print("Collision with wall")
            return True
        elif board_layout[cell_y][cell_x] == ".":
            print("Collision with point")
            self.score += 10
            board_layout[cell_y][cell_x] = " "
        elif board_layout[cell_y][cell_x] == "o":
            print("Collision with big point")
            self.score += 50
            board_layout[cell_y][cell_x] = " "
        elif board_layout[cell_y][cell_x] == "G":  
            print("Collision with ghost")
            self.score -= 100
            board_layout[cell_y][cell_x] = " "
        return False

    