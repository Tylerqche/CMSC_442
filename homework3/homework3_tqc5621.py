############################################################
# CMPSC 442: Homework 3
############################################################

student_name = "Tyler Cheng"

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
from collections import deque

############################################################
# Section 1: Sudoku
############################################################

def sudoku_cells():
    return [(r,c) for r in range(9) for c in range(9)]

def sudoku_arcs():
    arcs = []
    for cell in sudoku_cells():
        r, c = cell
        peers = set()
        # Same row
        for col in range(9):
            if col != c:
                peers.add((r, col))
        # Same column
        for row in range(9):
            if row != r:
                peers.add((row, c))
        # Same block
        block_r = r // 3
        block_c = c // 3
        for br in range(block_r * 3, block_r * 3 + 3):
            for bc in range(block_c * 3, block_c * 3 + 3):
                if br != r or bc != c:
                    peers.add((br, bc))
        # Add all arcs from cell to its peers
        for peer in peers:
            arcs.append((cell, peer))
    return arcs

def read_board(path):
    board = {}
    with open(path, 'r') as f:
        for row, line in enumerate(f):
            line = line.strip()
            if len(line) != 9:
                raise ValueError("Each line must contain exactly 9 characters")
            for col, char in enumerate(line):
                if char == '*':
                    board[(row, col)] = set(range(1, 10))
                else:
                    board[(row, col)] = {int(char)}
    return board

class Sudoku(object):

    CELLS = sudoku_cells()
    ARCS = sudoku_arcs()

    def __init__(self, board):
        self.board = {cell: set(values) for cell, values in board.items()}
        self.peers = {}
        for cell in self.CELLS:
            r, c = cell
            peers = set()
            # Same row
            for col in range(9):
                if col != c:
                    peers.add((r, col))
            # Same column
            for row in range(9):
                if row != r:
                    peers.add((row, c))
            # Same block
            block_r = r // 3
            block_c = c // 3
            for br in range(block_r * 3, block_r * 3 + 3):
                for bc in range(block_c * 3, block_c * 3 + 3):
                    if (br, bc) != (r, c):
                        peers.add((br, bc))
            self.peers[cell] = peers

    def get_values(self, cell):
        return self.board[cell]

    def remove_inconsistent_values(self, cell1, cell2):
        removed = False
        values1 = self.board[cell1]
        values2 = self.board[cell2]
        for x in list(values1):
            conflict = True
            for y in values2:
                if x != y:
                    conflict = False
                    break
            if conflict:
                values1.remove(x)
                removed = True
        return removed

    def infer_ac3(self):
        queue = deque(self.ARCS)
        while queue:
            cell1, cell2 = queue.popleft()
            if self.remove_inconsistent_values(cell1, cell2):
                if not self.board[cell1]:
                    return  # Contradiction
                for peer in self.peers[cell1]:
                    if peer != cell2:
                        queue.append((peer, cell1))

    def infer_improved(self):
        self.infer_ac3()
        while True:
            updated = False
            for cell in self.CELLS:
                values = self.board[cell]
                if len(values) == 1:
                    continue
                # Check for hidden singles in row, column, block
                row, col = cell
                # Check row
                for v in values:
                    if all(v not in self.board[(row, c)] for c in range(9) if (row, c) != cell):
                        self.board[cell] = {v}
                        updated = True
                        break
                if updated:
                    break
                # Check column
                for v in values:
                    if all(v not in self.board[(r, col)] for r in range(9) if (r, col) != cell):
                        self.board[cell] = {v}
                        updated = True
                        break
                if updated:
                    break
                # Check block
                block_r = row // 3
                block_c = col // 3
                block_cells = [(r, c) for r in range(block_r*3, block_r*3+3) for c in range(block_c*3, block_c*3+3) if (r, c) != cell]
                for v in values:
                    if all(v not in self.board[(r, c)] for (r, c) in block_cells):
                        self.board[cell] = {v}
                        updated = True
                        break
                if updated:
                    break
            if not updated:
                break
            self.infer_ac3()

    def infer_with_guessing(self):
        self.infer_improved()
        if all(len(v) == 1 for v in self.board.values()):
            return
        # Check for contradictions
        for cell in self.CELLS:
            if len(self.board[cell]) == 0:
                raise ValueError("Contradiction")
        # Find cell with minimum remaining values
        min_cell = None
        min_len = 10
        for cell in self.CELLS:
            l = len(self.board[cell])
            if l > 1 and l < min_len:
                min_len = l
                min_cell = cell
                if min_len == 2:
                    break
        # Try each possible value
        for value in sorted(self.board[min_cell]):
            new_board = {cell: set(values) for cell, values in self.board.items()}
            new_board[min_cell] = {value}
            new_sudoku = Sudoku(new_board)
            try:
                new_sudoku.infer_with_guessing()
                self.board = new_sudoku.board
                return
            except ValueError:
                pass
        raise ValueError("No solution")

