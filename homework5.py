############################################################
# CIS 521: Homework 5
############################################################

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import collections
import copy
import itertools
import random
import math

############################################################

student_name = "Linxi Wu"

############################################################
# Sudoku Solver
############################################################


def sudoku_cells():
    return [(row, col) for row in range(9) for col in range(9)]


def sudoku_arcs():
    arcs = []
    for cell1 in sudoku_cells():
        row1, col1 = cell1
        for cell2 in sudoku_cells():
            row2, col2 = cell2
            same_row = row1 == row2
            same_col = col1 == col2
            same_box = row1 // 3 == row2 // 3 and col1 // 3 == col2 // 3
            if cell1 != cell2 and (same_row or same_col or same_box):
                arcs.append((cell1, cell2))
    return arcs


def read_board(path):
    board = {}
    with open(path, "r") as file:
        rows = [line.strip() for line in file if line.strip()]
    if len(rows) != 9 or any(len(row) != 9 for row in rows):
        raise ValueError(
            "A Sudoku board must contain nine rows of nine values")
    for row, values in enumerate(rows):
        for col, value in enumerate(values):
            if value == "*" or value == "0":
                board[(row, col)] = set(range(1, 10))
            elif value in "123456789":
                board[(row, col)] = {int(value)}
            else:
                raise ValueError("Invalid Sudoku value")
    return board


class Sudoku(object):

    CELLS = sudoku_cells()
    ARCS = sudoku_arcs()

    def __init__(self, board):
        self.board = {cell: set(values) for cell, values in board.items()}

    def get_values(self, cell):
        return self.board[cell]

    def remove_inconsistent_values(self, cell1, cell2):
        removed = False
        values1 = self.board[cell1]
        values2 = self.board[cell2]
        if len(values2) == 1:
            value2 = next(iter(values2))
            if value2 in values1 and len(values1) > 1:
                values1.remove(value2)
                removed = True
        return removed

    def infer_ac3(self):
        queue = collections.deque(self.ARCS)
        while queue:
            cell1, cell2 = queue.popleft()
            if self.remove_inconsistent_values(cell1, cell2):
                if not self.board[cell1]:
                    return
                for neighbor in self._neighbors(cell1):
                    if neighbor != cell2:
                        queue.append((neighbor, cell1))

    def infer_improved(self):
        # Alternate AC-3 with the Sudoku-only-choice rule until stable.
        changed = True
        while changed:
            before = self._state()
            self.infer_ac3()
            for unit in self._units():
                for value in range(1, 10):
                    places = [cell for cell in unit
                              if value in self.board[cell]]
                    if len(places) == 1:
                        self.board[places[0]] = {value}
            changed = before != self._state()

    def infer_with_guessing(self):
        self.infer_improved()
        if any(not values for values in self.board.values()):
            return
        if all(len(values) == 1 for values in self.board.values()):
            return
        cell = min((c for c in self.CELLS if len(self.board[c]) > 1),
                   key=lambda c: len(self.board[c]))
        original = copy.deepcopy(self.board)
        for value in sorted(original[cell]):
            self.board = copy.deepcopy(original)
            self.board[cell] = {value}
            self.infer_with_guessing()
            if (all(self.board[c] for c in self.CELLS)
                    and self._is_valid_solution()):
                return
        self.board = original

    def _neighbors(self, cell):
        return {other for first, other in self.ARCS if first == cell}

    def _units(self):
        rows = [[(r, c) for c in range(9)] for r in range(9)]
        cols = [[(r, c) for r in range(9)] for c in range(9)]
        boxes = [[(r, c) for r in range(br, br + 3)
                  for c in range(bc, bc + 3)]
                 for br in (0, 3, 6) for bc in (0, 3, 6)]
        return rows + cols + boxes

    def _state(self):
        return tuple((cell, tuple(sorted(self.board[cell])))
                     for cell in self.CELLS)

    def _is_valid_solution(self):
        if any(len(self.board[cell]) != 1 for cell in self.CELLS):
            return False
        return all(len({next(iter(self.board[cell])) for cell in unit}) == 9
                   for unit in self._units())

############################################################
# Feedback
############################################################


# Just an approximation is fine.
feedback_question_1 = """
    The most challenging part was understanding how AC-3 propagates a change
    through the network of Sudoku constraints.  In particular, when a cell
    is reduced to one value, every neighboring cell must be reconsidered.
"""

feedback_question_2 = """
    I found the assignment helpful because it connected the abstract CSP ideas
    from class to a concrete problem.
"""

feedback_question_3 = """
No comments on improvement
"""
