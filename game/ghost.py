import pygame

class Ghost:
    def __init__(self, position: list, speed: float, size: int, color: tuple, direction: str = None):
        """
        Initialize the Ghost object.
        """
        self.position = [position[0], position[1]]  
        self.speed = speed
        self.size = size
        self.color = color
        self.direction = direction
    
    def draw(self, screen):
        """
        Draw the ghost on the screen.
        """
        pygame.draw.rect(screen, self.color, ((self.position[0], self.position[1]), (self.size, self.size)))

    def move(self):
        """
        Move the ghost in the selected direction.
        """

        
        cell_x = int(self.position[0] // self.size)
        cell_y = int(self.position[1] // self.size)

    def select_direction():
        pass
