############################################################
# CMPSC 442: Informed Search
############################################################

student_name = "Tyler Cheng"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import heapq
import random
from queue import PriorityQueue

############################################################
# Section 1: Tile Puzzle
############################################################

def create_tile_puzzle(rows, cols):
    board = []
    num = 1
    for i in range(rows):
        row = []
        for j in range(cols):
            if num < rows * cols:
                row.append(num)
                num += 1
            else:
                row.append(0)  
        board.append(row)
    return TilePuzzle(board)

class TilePuzzle(object):
    
    # Required
    def __init__(self, board):
        self.board = [list(row) for row in board]
        self.rows = len(board)
        self.cols = len(board[0]) if self.rows > 0 else 0
        self.empty_pos = None
        for i in range(self.rows):
            for j in range(self.cols):
                if self.board[i][j] == 0:
                    self.empty_pos = (i, j)

    def get_board(self):
        return [list(row) for row in self.board]

    def perform_move(self, direction):
        dirs = {
            "up": (-1, 0),
            "down": (1, 0),
            "left": (0, -1),
            "right": (0, 1)
        }
        if direction not in dirs:
            return False
        dr, dc = dirs[direction]
        r, c = self.empty_pos
        new_r, new_c = r + dr, c + dc
        if 0 <= new_r < self.rows and 0 <= new_c < self.cols:
            self.board[r][c], self.board[new_r][new_c] = self.board[new_r][new_c], self.board[r][c]
            self.empty_pos = (new_r, new_c)
            return True
        return False

    def scramble(self, num_moves):
        moves = ["up", "down", "left", "right"]
        for _ in range(num_moves):
            self.perform_move(random.choice(moves))


    def is_solved(self):
        solved = create_tile_puzzle(self.rows, self.cols).get_board()
        return self.board == solved

    def copy(self):
        return TilePuzzle([list(row) for row in self.board])

    def successors(self):
        moves = ["up", "down", "left", "right"]
        for move in moves:
            copy_puzzle = self.copy()
            if copy_puzzle.perform_move(move):
                yield (move, copy_puzzle)

    # Required
    def find_solutions_iddfs(self):
        depth = 0
        solutions = []
        while not solutions:
            for sol in self.iddfs_helper(depth, []):
                solutions.append(sol)
            depth += 1
        for sol in solutions:
            yield sol

    def iddfs_helper(self, limit, moves):
        if self.is_solved():
            yield moves
            return
        if len(moves) >= limit:
            return
        for move, successor in self.successors():
            for solution in successor.iddfs_helper(limit, moves + [move]):
                yield solution

    # Required
    def find_solution_a_star(self):
        def manhattan_distance(board):
            total = 0
            rows, cols = len(board), len(board[0])
            for i in range(rows):
                for j in range(cols):
                    value = board[i][j]
                    if value != 0:
                        target_row = (value - 1) // cols
                        target_col = (value - 1) % cols
                        total += abs(target_row - i) + abs(target_col - j)
            return total

        visited = set()
        pq = PriorityQueue()
        start_h = manhattan_distance(self.board)

        # Priority queue stores (f_score, g_score, id(state), state, path)
        pq.put((start_h, 0, id(self), self, []))

        while not pq.empty():
            f, g, _, current, path = pq.get()
            board_tuple = tuple(tuple(row) for row in current.board)

            if current.is_solved():
                return path

            if board_tuple in visited:
                continue

            visited.add(board_tuple)

            for move, new_state in current.successors():
                new_board_tuple = tuple(tuple(row) for row in new_state.board)
                if new_board_tuple not in visited:
                    new_g = g + 1
                    new_h = manhattan_distance(new_state.board)
                    new_f = new_g + new_h
                    pq.put((new_f, new_g, id(new_state), new_state, path + [move]))

        return None
    
############################################################
# Section 2: Grid Navigation
############################################################

def find_path(start, goal, scene):
    if not (0 <= start[0] < len(scene) and 0 <= start[1] < len(scene[0])) or \
       not (0 <= goal[0] < len(scene) and 0 <= goal[1] < len(scene[0])) or \
       scene[start[0]][start[1]] or scene[goal[0]][goal[1]]:
        return None

    def euclidean_distance(pos1, pos2):
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

    queue = PriorityQueue()
    queue.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}

    while not queue.empty():
        current = queue.get()[1]
        
        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        for dx, dy in directions:
            next_pos = (current[0] + dx, current[1] + dy)
            
            if not (0 <= next_pos[0] < len(scene) and 0 <= next_pos[1] < len(scene[0])):
                continue
            if scene[next_pos[0]][next_pos[1]]:
                continue
                
            new_cost = cost_so_far[current] + euclidean_distance(current, next_pos)
            
            if next_pos not in cost_so_far or new_cost < cost_so_far[next_pos]:
                cost_so_far[next_pos] = new_cost
                priority = new_cost + euclidean_distance(next_pos, goal)
                queue.put((priority, next_pos))
                came_from[next_pos] = current

    return None

############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################

def solve_distinct_disks(length, n):
    initial_board = [-1] * length
    for i in range(n):
        initial_board[i] = i
    initial_state = tuple(initial_board)

    target_board = [-1] * length
    for i in range(n):
        target_board[length - n + i] = n - 1 - i
    target_state = tuple(target_board)
    
    def heuristic(state):
        total = 0
        for d in range(n):
            current_pos = state.index(d)
            target_pos = length - 1 - d
            dist = abs(target_pos - current_pos)
            total += (dist + 1) // 2 
        return total
    
    # Each entry is a tuple (priority, cost_so_far, state, moves_list)
    pq = [(heuristic(initial_state), 0, initial_state, [])]
    visited = {initial_state: 0}
    
    # A move is represented as a tuple (from_pos, to_pos).
    def get_valid_moves(state):
        valid_moves = []
        for pos in range(length):
            if state[pos] != -1:  # there is a disk at pos
                # Check adjacent moves
                for next_pos in (pos - 1, pos + 1):
                    if 0 <= next_pos < length and state[next_pos] == -1:
                        valid_moves.append((pos, next_pos))
                # Check jump moves (move two cells) provided the intermediate cell is occupied
                for next_pos in (pos - 2, pos + 2):
                    if 0 <= next_pos < length and state[next_pos] == -1:
                        middle_pos = (pos + next_pos) // 2
                        if state[middle_pos] != -1:
                            valid_moves.append((pos, next_pos))
        return valid_moves
    
    # Applies a move to a state
    def make_move(state, from_pos, to_pos):
        new_state = list(state)
        new_state[to_pos] = new_state[from_pos]
        new_state[from_pos] = -1
        return tuple(new_state)
    
    # A* search loop
    while pq:
        priority, cost, current_state, moves = heapq.heappop(pq)
        
        if current_state == target_state:
            return moves
        
        for from_pos, to_pos in get_valid_moves(current_state):
            new_state = make_move(current_state, from_pos, to_pos)
            new_cost = cost + 1
            if new_state not in visited or new_cost < visited[new_state]:
                visited[new_state] = new_cost
                heapq.heappush(pq, (new_cost + heuristic(new_state), new_cost, new_state, moves + [(from_pos, to_pos)]))
    
    return None

############################################################
# Section 4: Dominoes Game
############################################################

def create_dominoes_game(rows, cols):
    board = [[False for _ in range(cols)] for _ in range(rows)]
    return DominoesGame(board)

class DominoesGame(object):

    # Required
    def __init__(self, board):
        self.board = [list(row) for row in board]
        self.rows = len(board)
        self.cols = len(board[0]) if self.rows > 0 else 0

    def get_board(self):
        return [list(row) for row in self.board]

    def reset(self):
        self.board = [[False for _ in range(self.cols)] for _ in range(self.rows)]

    def is_legal_move(self, row, col, vertical):
        if vertical:
            return (row + 1 < self.rows and 
                    not self.board[row][col] and 
                    not self.board[row + 1][col])
        else:
            return (col + 1 < self.cols and 
                    not self.board[row][col] and 
                    not self.board[row][col + 1])

    def legal_moves(self, vertical):
        moves = []
        if vertical:
            for row in range(self.rows - 1):
                for col in range(self.cols):
                    if self.is_legal_move(row, col, vertical):
                        moves.append((row, col))
        else:
            for row in range(self.rows):
                for col in range(self.cols - 1):
                    if self.is_legal_move(row, col, vertical):
                        moves.append((row, col))
        return moves

    def perform_move(self, row, col, vertical):
        if self.is_legal_move(row, col, vertical):
            if vertical:
                self.board[row][col] = True
                self.board[row + 1][col] = True
            else:
                self.board[row][col] = True
                self.board[row][col + 1] = True

    def game_over(self, vertical):
        return len(self.legal_moves(vertical)) == 0

    def copy(self):
        return DominoesGame([list(row) for row in self.board])

    def successors(self, vertical):
        for move in self.legal_moves(vertical):
            successor = self.copy()
            successor.perform_move(move[0], move[1], vertical)
            yield (move, successor)

    def get_random_move(self, vertical):
        moves = self.legal_moves(vertical)
        return random.choice(moves) if moves else None

    # Required
    def get_best_move(self, vertical, limit):
        root = vertical

        def evaluate(game):
            return len(game.legal_moves(root)) - len(game.legal_moves(not root))
        
        # Standard alpha-beta search that returns (value, leaves_visited)
        def alphabeta(game, depth, alpha, beta, maximizing, current_player):
            if depth == 0 or game.game_over(current_player):
                return evaluate(game), 1

            total_leaves = 0
            if maximizing:
                value = float('-inf')
                for move, successor in game.successors(current_player):
                    child_value, child_leaves = alphabeta(successor, depth - 1, alpha, beta, False, not current_player)
                    total_leaves += child_leaves
                    value = max(value, child_value)
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break  # beta cutoff
                return value, total_leaves
            else:
                value = float('inf')
                for move, successor in game.successors(current_player):
                    child_value, child_leaves = alphabeta(successor, depth - 1, alpha, beta, True, not current_player)
                    total_leaves += child_leaves
                    value = min(value, child_value)
                    beta = min(beta, value)
                    if alpha >= beta:
                        break  # alpha cutoff
                return value, total_leaves

        best_move = None
        best_value = float('-inf')
        total_leaves = 0
        alpha = float('-inf')
        beta = float('inf')
        # Unified search at the root: iterate through all successors and update alpha/beta across them
        for move, successor in self.successors(root):
            child_value, child_leaves = alphabeta(successor, limit - 1, alpha, beta, False, not root)
            total_leaves += child_leaves
            if child_value > best_value:
                best_value = child_value
                best_move = move
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break
        return best_move, best_value, total_leaves
