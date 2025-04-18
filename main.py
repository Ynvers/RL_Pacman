import game.player as player
import game.ghost as ghost
import pygame
from game.board import board_layout, Board
import time

#Pygame setup
pygame.init()

# Set up the game window
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 450
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
player_start_position = board.get_player_start_position()
if player_start_position is None:
    raise ValueError("Player start position not found in board layout.")
else:
    print("player start position found: ", player_start_position)
    pacman = player.Player(player_start_position, 4, CELL_SIZE, [])

# Initialisation du timer
start_time = time.time()
game_duration = 60  # 60 secondes
total_points = sum(row.count('.') + row.count('o') for row in board.layout)


# Main loop
running = True
game_over = False
font = pygame.font.Font(None, 36)
clock = pygame.time.Clock()

while running:
    #print("Boucle principale en cours...")

    current_time = time.time()
    elapsed_time = current_time - start_time
    remaining_time = max(0, game_duration - elapsed_time)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if not game_over:
        if remaining_time <= 0 or pacman.score >= total_points:
            game_over = True
            if pacman.score >= total_points:
                pacman.game_won = True
                print("Vous avez gagné !")
            else:
                print("Temps écoulé ! Vous avez perdu.")
            break
    #keyboard events
    keys = pygame.key.get_pressed()
    if keys[pygame.K_a] or keys[pygame.K_UP]:
        pacman.move("UP", board.layout)
        pacman.check_collision(board.layout)
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        pacman.move("DOWN", board.layout)
        pacman.check_collision(board.layout)
    if keys[pygame.K_q] or keys[pygame.K_LEFT]:
        pacman.move("LEFT", board.layout)
        pacman.check_collision(board.layout)
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        pacman.move("RIGHT", board.layout)
        pacman.check_collision(board.layout)

    # Fill the screen with black
    screen.fill(BLACk)

    # Dessiner la grille
    board.draw(screen)

    # Draw the player
    pacman.draw(screen)

    # Afficher le timer et le score
    timer_text = font.render(f"Time: {int(remaining_time)}s", True, WHITE)
    #score_text = font.render(f"Score: {pacman.score}", True, WHITE)
    points_text = font.render(f"Points: {pacman.score}/{total_points}", True, WHITE)
    
    screen.blit(timer_text, (10, 10))
    #screen.blit(score_text, (10, 40))
    screen.blit(points_text, (10, 70))

    # Afficher le message de fin de jeu si nécessaire
    if game_over:
        game_over_text = font.render(
            "YOU WIN!" if pacman.game_won else "GAME OVER!", 
            True, 
            WHITE
        )
        text_rect = game_over_text.get_rect(center=(SCREEN_WIDTH/2, SCREEN_HEIGHT/2))
        screen.blit(game_over_text, text_rect)

    # Update the display
    pygame.display.flip()

pygame.quit()
print("Fin du jeu")