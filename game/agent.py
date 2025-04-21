import random
import pickle

class Agent:
    def __init__(self):
        self.q_table = {}
        self.epsilon = 1.0  # Commencer à 1.0 pour plus d'exploration
        self.epsilon_min = 0.01  # Valeur minimale d'epsilon
        self.alpha = 0.1  # Learning rate
        self.epsilon_decay = 0.995  # Decay plus lent
        self.gamma = 0.99  # Augmenter gamma pour donner plus d'importance aux récompenses futures
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]
        self.previous_point_dist = float('inf')  # Ajouter cette ligne

    def get_state(self, pacman_pos, ghost_pos, board_layout):
        """
        Convert the game state into a tuple for the Q-table.
        Return a tuplecontaining:
        - Pacman position (x, y)
        - Distance to the nearest ghost
        - Direction to the nearest ghost
        - Distance to the nearest point
        """
        def bfs_distance(start, targets, board_layout):
            """
            It's a helper function to calculate the distance from the start position using the walls to the targets using BFS.
            """
            to_visit = [(start, 0)]
            visited = set()
            visited.add(start)

            while to_visit:
                (x, y), dist = to_visit.pop(0)
                
                # Check if the current position is a target
                if (x, y) in targets:
                    return dist, (x, y)
                
                #Check up for the 04 directions
                for dx, dy, direction in [(0, -1, "UP"), (0, 1, "DOWN"), (-1, 0, "LEFT"), (1, 0, "RIGHT")]:
                    new_x, new_y = x + dx, y + dy
                    if (0 <= new_x < len(board_layout[0]) and
                        0 <= new_y < len(board_layout) and
                        board_layout[new_y][new_x] != "#"
                        and (new_x, new_y) not in visited):
                        to_visit.append(((new_x, new_y), dist + 1))
                        visited.add((new_x, new_y))

            return float('inf'), None  # No path found

        # Convert pacman's positions to grid coordinates
        pacman_cell = (int(pacman_pos[0] // 20), int(pacman_pos[1] // 20))
        
        # Get ghost's positions in grid coordinates
        ghost_cells = [(int(pos[0] // 20), int(pos[1] // 20)) for pos in ghost_pos]

        # Find the nearest ghost and it's direction
        ghost_dist, nearest_ghost = bfs_distance(pacman_cell, ghost_cells, board_layout)

        # Determine direction to the nearest ghost
        ghost_direction = None
        if nearest_ghost:
            dx = nearest_ghost[0] - pacman_cell[0]
            dy = nearest_ghost[1] - pacman_cell[1]
            if abs(dx) > abs(dy):
                ghost_direction = "LEFT" if dx < 0 else "RIGHT"
            else:
                ghost_direction = "UP" if dy < 0 else "DOWN"

        # Get the distance to the nearest point (food)
        dots = set()
        for y in range(len(board_layout)):
            for x in range(len(board_layout[y])):
                if board_layout[y][x] == ".":
                    dots.add((x, y))
        point_dist, nearest_point = bfs_distance(pacman_cell, dots, board_layout)

        # Return the state as a tuple
        return (
            pacman_cell[0],  # Pacman's x position
            pacman_cell[1],  # Pacman's y position
            ghost_dist,      # Distance to the nearest ghost
            ghost_direction, # Direction to the nearest ghost
            point_dist,      # Distance to the nearest point (food)
            nearest_point     # Position of the nearest point (food)
        )

    def get_valid_actions(self, state, board_layout):
        """
        Returns a list of valid actions based on the current state.
        """
        x, y = state[0], state[1] # Pacman's position
        valid_actions = []

        # Check for each direction
        for dx, dy, action in [(0, -1, "UP"), (0, 1, "DOWN"), (-1, 0, "LEFT"), (1, 0, "RIGHT")]:
            new_x = x + dx
            new_y = y + dy
            if (0 <= new_x < len(board_layout[0]) and 
                0 <= new_y < len(board_layout) and 
                board_layout[new_y][new_x] != "#"):
                valid_actions.append(action)

        return valid_actions

    def choose_action(self, state, board_layout):
        """
        Chooses an action using e-greedy policy.
        """
        # Exploration
        if random.random() < self.epsilon:
            valid_actions = self.get_valid_actions(state, board_layout)
            return random.choice(valid_actions) if valid_actions else None
        
        # Exploitation
        return self.get_best_action(state, board_layout)

    def get_best_action(self, state, board_layout):
        """
        Returns the best action based on the current state.
        """
        valid_actions = self.get_valid_actions(state, board_layout) 
        if not valid_actions:
            return None
        
        # Initialize state in Q-table if not present
        if state not in self.q_table:
            self.q_table[state] = {action: 0 for action in self.actions}

        # Get the action with the maximum Q-value
        return max(valid_actions, key=lambda action: self.q_table[state].get(action, 0))
    
    def learn(self, state, action, reward, next_state):
        """Mise à jour de la Q-table avec des récompenses ajustées"""
        if state not in self.q_table:
            self.q_table[state] = {a: 0 for a in self.actions}
        
        if next_state not in self.q_table:
            self.q_table[next_state] = {a: 0 for a in self.actions}

        # Ajuster les récompenses basées sur la distance aux fantômes
        ghost_dist = state[2]
        if ghost_dist < 3:  # Danger immédiat
            reward -= 50
        elif ghost_dist > 5:  # Zone sûre
            reward += 5

        # Récompense pour se rapprocher des points
        point_dist = state[4]
        if point_dist < self.previous_point_dist:  # Utiliser self.previous_point_dist
            reward += 10
        
        self.previous_point_dist = point_dist  # Mettre à jour pour la prochaine fois

        # Q-learning update
        current_q = self.q_table[state].get(action, 0)
        next_max_q = max(self.q_table[next_state].values())
        new_q = current_q + self.alpha * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state][action] = new_q

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_qtable(self, filename='qtable.pkl'):
        """Sauvegarde la Q-table"""
        with open(filename, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_qtable(self, filename='qtable.pkl'):
        """Charge la Q-table"""
        try:
            with open(filename, 'rb') as f:
                self.q_table = pickle.load(f)
        except FileNotFoundError:
            print("Pas de Q-table sauvegardée trouvée")