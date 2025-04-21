import game.player as player
import game.ghost as Ghost
import pygame
from game.board import board_layout, Board
from game.agent import Agent
import time
import matplotlib.pyplot as plt
import numpy as np

#Pygame setup
pygame.init()

# Set up the game window
SCREEN_WIDTH = 565
SCREEN_HEIGHT = 430

# Set up the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Pacman Game - RL Training")

#Set up the color
BLACk = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

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

# Initialise the ghosts
ghosts_start_positions = board.get_ghost_start_position()
if ghosts_start_positions is None:
    raise ValueError("Ghosts start position not found in board layout.")
else:
    print("ghosts start position found: ", ghosts_start_positions)
    ghosts = [
        Ghost.Ghost(position, 2, CELL_SIZE, RED, "BFS") for position in ghosts_start_positions
    ]
    print("ghosts created")

# Initialisation the timer
start_time = time.time()
game_duration = 60  # 60 secondes
total_points = sum(row.count('.') + row.count('o') * 5 for row in board.layout)

# Initialisation de l'agent
agent = Agent()

# Add these variables to track the performance of the agent
episode = 0
max_episodes = 5
scores = []
previous_score = 0
epsilons = []

def plot_metrics(scores, epsilons):
    episodes = range(len(scores))
    
    # Plot des scores
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(episodes, scores)
    plt.title('Score par épisode')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    # Plot de epsilon
    plt.subplot(1, 2, 2)
    plt.plot(episodes, epsilons)
    plt.title('Epsilon par épisode')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()

# Main loop
running = True
game_over = False
font = pygame.font.Font(None, 36)
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    current_time = time.time()
    elapsed_time = current_time - start_time
    remaining_time = max(0, game_duration - elapsed_time)

    # Fill the screen with black UNE SEULE FOIS au début
    screen.fill(BLACk)

    if not game_over:
        # Logique de l'agent et du jeu
        # Vérifier si tous les points sont collectés au lieu du timer
        if len([cell for row in board.layout for cell in row if cell in ['.', 'o']]) == 0:
            game_over = True
            pacman.game_won = True
            print("Victoire ! Tous les points ont été collectés !")
            continue

        # Agent logic
        current_state = agent.get_state(
            pacman.position, 
            [ghost.position for ghost in ghosts],
            board.layout
        )
        
        action = agent.choose_action(current_state, board.layout)
        if action:
            # Sauvegarder l'état avant le mouvement
            previous_score = pacman.score
            previous_pos = pacman.position.copy()
            
            # Effectuer le mouvement
            pacman.move(action, board.layout)
            pacman.check_collision(board.layout)  # Important: ajoutez cette ligne!
            
            # Calculer la récompense
            reward = 0
            
            # Récompense pour manger des points
            if pacman.score > previous_score:
                reward += 10  # Récompense pour chaque point mangé
            
            # Pénalité pour rester sur place
            if pacman.position == previous_pos:
                reward -= 1
            
            # Mettre à jour l'agent avec le nouvel état
            next_state = agent.get_state(
                pacman.position,
                [ghost.position for ghost in ghosts],
                board.layout
            )
            agent.learn(current_state, action, reward, next_state)
            
            # Dans la boucle principale, après le mouvement
            if action:
                print(f"Action: {action}")
                print(f"Position: {pacman.position}")
                print(f"Score: {pacman.score}")
                print(f"Reward: {reward}")
                print("---")
        
        # Ghost movement and collision check
        for ghost in ghosts:
            ghost.move(board.layout, pacman.position)
            if ghost.check_collision_with_pacman(pacman.position):
                reward = -100  # Collision avec fantôme
                game_over = True
                pacman.game_won = False
                print("Game Over! Collision avec un fantôme!")
                break

        # Render only if game is still active
        board.draw(screen)
        pacman.draw(screen)
        for ghost in ghosts:
            ghost.draw(screen)

        # UI elements
        timer_text = font.render(f"Time: {int(remaining_time)}s", True, WHITE)
        points_text = font.render(f"Points: {pacman.score}/{total_points}", True, WHITE)
        screen.blit(timer_text, (10, 10))
        screen.blit(points_text, (10, 70))

    # Update display ONLY ONCE per frame
    pygame.display.flip()
    clock.tick(60)

    if game_over:
        episode += 1
        scores.append(pacman.score)
        epsilons.append(agent.epsilon)
        
        # Affichage plus concis des métriques
        if episode % 10 == 0:  # Afficher seulement tous les 10 épisodes
            print(f"Episode {episode}/{max_episodes}, Score: {pacman.score}, Epsilon: {agent.epsilon:.4f}")
        
        if episode >= max_episodes:  # Changé de < à >=
            print("\nEntraînement terminé!")
            print(f"Score moyen: {np.mean(scores):.2f}")
            print(f"Score maximum: {max(scores)}")
            # Tracer les graphiques
            plot_metrics(scores, epsilons)
            running = False  # Arrêter la boucle principale
            break
        
        # Reset game state without rendering
        game_over = False
        start_time = time.time()
        pacman.reset(player_start_position.copy())
        board.reset()
        for ghost, start_pos in zip(ghosts, ghosts_start_positions):
            ghost.position = start_pos.copy()
        previous_score = 0
        remaining_time = game_duration
        continue

# Afficher le résultat final
if game_over:
    if pacman.game_won:
        print("Félicitations! Vous avez gagné!")
    else:
        print("Game Over! Un fantôme vous a attrapé!")

pygame.quit()
print("Fin du jeu")