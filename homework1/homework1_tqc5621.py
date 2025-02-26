############################################################
# CMPSC 442: Uninformed Search
############################################################

student_name = "Tyler Cheng"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import math
import random
from collections import deque

############################################################
# Section 1: N-Queens
############################################################

def num_placements_all(n):
    return 2 ** (n * n)

def num_placements_one_per_row(n):
    return n ** n

def n_queens_valid(board):
    n = len(board)
    for i in range(n):
        for j in range(i + 1, n):
            # Check if two queens are in the same column or on the same diagonal
            if board[i] == board[j] or abs(board[i] - board[j]) == abs(i - j):
                return False
    return True

def n_queens_solutions(n):

    def n_queens_helper(n, board):
        if len(board) == n:
            yield board
            return

        for col in range(n):
            new_board = board + [col]  
            if n_queens_valid(new_board):
                yield from n_queens_helper(n, new_board)
    
    yield from n_queens_helper(n, [])

############################################################
# Section 2: Lights Out
############################################################

class LightsOutPuzzle(object):

    def __init__(self, board):
        self.board = board
        self.rows = len(board)
        self.cols = len(board[0]) if self.rows > 0 else 0

    def get_board(self):
        return self.board

    def perform_move(self, row, col):
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        self.board[row][col] = not self.board[row][col]
        for dx, dy in directions:
            next_row, next_col = row + dx, col + dy
            if 0 <= next_row < self.rows and 0 <= next_col < self.cols:
                self.board[next_row][next_col] = not self.board[next_row][next_col]
        
    def scramble(self):
        for row in range(self.rows):
            for col in range(self.cols):
                if random.random() < 0.5:
                    self.perform_move(row, col)

    def is_solved(self):
        for row in range(self.rows):
            for col in range(self.cols): 
                if self.board[row][col] == True:
                    return False
        return True

    def copy(self):
        return LightsOutPuzzle([row.copy() for row in self.board])

    def successors(self):
        for r in range(self.rows):
            for c in range(self.cols):
                new_puzzle = self.copy()
                new_puzzle.perform_move(r, c)
                yield (r, c), new_puzzle

    def find_solution(self):
        queue = deque([(self, [])])
        visited = set()

        while queue:
            current_puzzle, moves = queue.popleft()
            if current_puzzle.is_solved():
                return moves

            board_tuple = tuple(tuple(row) for row in current_puzzle.get_board())
            if board_tuple in visited:
                continue

            visited.add(board_tuple)
            for move, new_puzzle in current_puzzle.successors():
                queue.append((new_puzzle, moves + [move]))

        return None
    
def create_puzzle(rows, cols):
    board = [[False] * cols for _ in range(rows)]
    return LightsOutPuzzle(board)

############################################################
# Section 3: Linear Disk Movement
############################################################

def solve_identical_disks(length, n):
    initial_positions = tuple(sorted(range(n)))  
    target_positions = tuple(sorted(range(length - n, length))) 
    
    queue = deque([(initial_positions, [])])
    visited = {initial_positions}
    
    def get_valid_moves(positions):
        disk_positions = set(positions)
        valid_moves = []

        for pos in positions:
            # Move to adjacent empty cell
            for next_pos in (pos - 1, pos + 1):
                if 0 <= next_pos < length and next_pos not in disk_positions:
                    valid_moves.append((pos, next_pos))
            
            # Move two cells away if there's a disk in between
            for next_pos in (pos - 2, pos + 2):
                if 0 <= next_pos < length and next_pos not in disk_positions:
                    middle_pos = (pos + next_pos) // 2
                    if middle_pos in disk_positions:
                        valid_moves.append((pos, next_pos))
        
        return valid_moves
    
    while queue:
        current_positions, moves = queue.popleft()
        
        if current_positions == target_positions:
            return moves
        
        for from_pos, to_pos in get_valid_moves(current_positions):

            new_positions = list(current_positions)
            new_positions.remove(from_pos)
            new_positions.append(to_pos)
            new_positions = tuple(sorted(new_positions))  # Sort for identical disks
            
            if new_positions not in visited:
                visited.add(new_positions)
                queue.append((new_positions, moves + [(from_pos, to_pos)]))
    
    return None

def solve_distinct_disks(length, n):

    # Initial State
    initial_board = [-1] * length
    for i in range(n):
        initial_board[i] = i
    initial_state = tuple(initial_board)
    
    # Target State
    target_board = [-1] * length
    for i in range(n):
        target_board[length - n + i] = n - 1 - i
    target_state = tuple(target_board)
    
    # BFS queue storing (state, moves_list)
    queue = deque([(initial_state, [])])
    visited = {initial_state}
    
    def get_valid_moves(state):
        valid_moves = []
        
        for pos in range(length):
            if state[pos] != -1:  
                for next_pos in (pos - 1, pos + 1):
                    if 0 <= next_pos < length and state[next_pos] == -1:
                        valid_moves.append((pos, next_pos))
                
                for next_pos in (pos - 2, pos + 2):
                    if 0 <= next_pos < length and state[next_pos] == -1:
                        middle_pos = (pos + next_pos) // 2
                        if state[middle_pos] != -1:  
                            valid_moves.append((pos, next_pos))
        
        return valid_moves
    
    def make_move(state, from_pos, to_pos):
        new_state = list(state)
        new_state[to_pos] = new_state[from_pos]
        new_state[from_pos] = -1
        return tuple(new_state)
    
    while queue:
        current_state, moves = queue.popleft()
        
        if current_state == target_state:
            return moves
        
        for from_pos, to_pos in get_valid_moves(current_state):
            new_state = make_move(current_state, from_pos, to_pos)
            
            if new_state not in visited:
                visited.add(new_state)
                queue.append((new_state, moves + [(from_pos, to_pos)]))
    
    return None 
