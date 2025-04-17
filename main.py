import game.player as player
import game.ghost as ghost
import pygame
from game.board import board_layout, Board

#Pygame setup
pygame.init()

# Set up the game window
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pacman Game")

#Set up the color
BLACk = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)

#Set up the cell size
CELL_SIZE = 20

# Initialise the board
board = Board(board_layout, CELL_SIZE, BLUE, YELLOW)

# Initialise the player
#pacman = player.Player((100, 100), 5, 20, 0, [])

# Main loop
running = True

while running:
    #print("Boucle principale en cours...")
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Fill the screen with black
        screen.fill(BLACk)

        # Dessiner la grille
        board.draw(screen)

        # Update the display
        pygame.display.flip()

pygame.quit()
print("Fin du jeu")