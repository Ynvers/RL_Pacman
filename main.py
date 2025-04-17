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
pacman = player.Player([100, 100], 5, 10, 0, [])

# Main loop
running = True
clock = pygame.time.Clock()

while running:
    #print("Boucle principale en cours...")
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        #keyboard events
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a] or keys[pygame.K_UP]:
            pacman.move("UP")
            if pacman.check_collision(board.layout):
                pacman.move("DOWN")
        if keys[pygame.K_d] or keys[pygame.K_DOWN]:
            pacman.move("DOWN")
            if pacman.check_collision(board.layout):
                pacman.move("UP")
        if keys[pygame.K_q] or keys[pygame.K_LEFT]:
            pacman.move("LEFT")
            if pacman.check_collision(board.layout):
                pacman.move("RIGHT")
        if keys[pygame.K_s] or keys[pygame.K_RIGHT]:
            pacman.move("RIGHT")
            if pacman.check_collision(board.layout):
                pacman.move("LEFT")

        # Fill the screen with black
        screen.fill(BLACk)

        # Dessiner la grille
        board.draw(screen)

        # Draw the player
        pacman.draw(screen)

        # Update the display
        pygame.display.flip()

pygame.quit()
print("Fin du jeu")