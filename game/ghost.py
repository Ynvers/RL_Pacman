import pygame
import time

class Ghost:
    def __init__(self, position: list, speed: float, size: int, color: tuple, strategy: str = "DFS"):
        """
        Initialize the Ghost object.
        """
        self.position = [position[0], position[1]]  
        self.speed = speed
        self.size = size
        self.color = color
        self.strategy = strategy
        self.last_move_time = time.time()

    def get_neighbors(self, board_layout):
        """
        Get the the accessibles neighbors of the ghost.
        """
        # Convert the ghost's position to grid coordinates
        x, y = int(self.position[0] // self.size), int(self.position[1] // self.size)
        neighbors = []
        directions = [(0, -1, "UP"), (0, 1, "DOWN"), (-1, 0, "LEFT"), (1, 0, "RIGHT")]

        for dx, dy, direction in directions:
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < len(board_layout[0]) and 0 <= new_y < len(board_layout):
                if board_layout[new_y][new_x] != "#":
                    neighbors.append((new_x, new_y, direction))
        return neighbors
        
    def bfs_search(self, board_layout, target_position):
        """
        BFS search algorithm for the ghost.
        """
        # Convert positions to grid coordinates
        start = (int(self.position[0] // self.size), int(self.position[1] // self.size))
        target = (int(target_position[0] // self.size), int(target_position[1] // self.size))

        #print(f"Start: {start}, Target: {target}")  # Debug print

        # Initialize the queue and visited set
        to_visit = [(start, [])]  # (position, path)
        visited = set()

        while to_visit:
            current_position, path = to_visit.pop(0)
            
            if current_position == target:
                print(f"Path found: {path}")  # Debug print
                return path[0] if path else None

            if current_position not in visited:
                visited.add(current_position)
                
                # Get neighbors for current position
                x, y = current_position
                for dx, dy, direction in [(0, -1, "UP"), (0, 1, "DOWN"), (-1, 0, "LEFT"), (1, 0, "RIGHT")]:
                    new_x, new_y = x + dx, y + dy
                    if (0 <= new_x < len(board_layout[0]) and 
                        0 <= new_y < len(board_layout) and 
                        board_layout[new_y][new_x] != "#"):
                        next_pos = (new_x, new_y)
                        if next_pos not in visited:
                            new_path = path + [direction]
                            to_visit.append((next_pos, new_path))
                            #print(f"Added to queue: pos={next_pos}, path={new_path}")  # Debug print

        print("No path found")  # Debug print
        return None


    def dfs_search(self):
        """
        DFS search algorithm for the ghost.
        """

    def ucs_search(self):
        """
        UCS search algorithm for the ghost.
        """
        pass
    
    def draw(self, screen):
        """
        Draw the ghost on the screen.
        """
        pygame.draw.rect(screen, self.color, ((self.position[0], self.position[1]), (self.size, self.size)))

    def select_direction(self, board_layout, pacman_pos):
        """
        Move the ghost in the selected direction.
        """
        if self.strategy == "BFS":
            return self.bfs_search(board_layout, pacman_pos)
        elif self.strategy == "DFS":
            return self.dfs_search(board_layout, pacman_pos)
        elif self.strategy == "UCS":
            return self.ucs_search(board_layout, pacman_pos)
    
    def check_collision_with_pacman(self, pacman_pos):
        """
        Check if the ghost collides with Pacman.
        Returns True if there is a collision, False otherwise.
        """
        ghost_cell_x = int(self.position[0] // self.size)
        ghost_cell_y = int(self.position[1] // self.size)
        pacman_cell_x = int(pacman_pos[0] // self.size)
        pacman_cell_y = int(pacman_pos[1] // self.size)
        
        return ghost_cell_x == pacman_cell_x and ghost_cell_y == pacman_cell_y

    def move(self, board_layout, pacman_pos):
        """
        Select a direction with a research algorithm for the ghost.
        """
        current_time = time.time()
        if current_time - self.last_move_time < 1 / self.speed:
            return  # Trop tôt pour un nouveau mouvement

        self.last_move_time = current_time
        
        # Vérifier que les positions sont valides
        print(f"Ghost position: {self.position}, Pacman position: {pacman_pos}")  # Debug print
        
        direction = self.select_direction(board_layout, pacman_pos)
        print(f"Direction choisie : {direction}")  # Debug print
        
        if direction:
            old_pos = self.position.copy()
            new_pos = self.position.copy()
            
            if direction == "UP":
                new_pos[1] -= self.size
            elif direction == "DOWN":
                new_pos[1] += self.size
            elif direction == "LEFT":
                new_pos[0] -= self.size
            elif direction == "RIGHT":
                new_pos[0] += self.size
                
            # Vérifier que la nouvelle position est valide
            new_cell_x = int(new_pos[0] // self.size)
            new_cell_y = int(new_pos[1] // self.size)
            if (0 <= new_cell_x < len(board_layout[0]) and 
                0 <= new_cell_y < len(board_layout) and 
                board_layout[new_cell_y][new_cell_x] != "#"):
                self.position = new_pos
                print(f"Ghost moved from {old_pos} to {self.position}")  # Debug print