############################################################
# CIS 521: Homework 2
############################################################

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
import math
import random
from collections import deque

############################################################

student_name = "Linxi Wu"

############################################################
# Section 1: N-Queens
############################################################


def num_placements_all(n):
    return math.comb(n * n, n)


def num_placements_one_per_row(n):
    return n ** n


def n_queens_valid(board):
    cols = set()
    diag1 = set()
    diag2 = set()

    for row, col in enumerate(board):
        if col in cols:
            return False
        diag_main = row - col
        diag_anti = row + col
        if diag_main in diag1 or diag_anti in diag2:
            return False
        cols.add(col)
        diag1.add(diag_main)
        diag2.add(diag_anti)

    return True


def n_queens_solutions(n):
    solutions = []

    def helper(board):
        if len(board) == n:
            solutions.append(list(board))
            return

        for col in range(n):
            board.append(col)
            if n_queens_valid(board):
                helper(board)
            board.pop()

    helper([])
    return solutions

############################################################
# Section 2: Lights Out
############################################################


class LightsOutPuzzle(object):

    def __init__(self, board):
        self.__board = [list(row) for row in board]
        self.__rows = len(self.__board)
        self.__cols = len(self.__board[0]) if self.__rows else 0

    def get_board(self):
        return [list(row) for row in self.__board]

    def perform_move(self, row, col):
        if not (0 <= row < self.__rows and 0 <= col < self.__cols):
            return

        self.__board[row][col] = not self.__board[row][col]
        for d_row, d_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = row + d_row
            nc = col + d_col
            if 0 <= nr < self.__rows and 0 <= nc < self.__cols:
                self.__board[nr][nc] = not self.__board[nr][nc]

    def scramble(self):
        for row in range(self.__rows):
            for col in range(self.__cols):
                if random.random() < 0.5:
                    self.perform_move(row, col)

    def is_solved(self):
        return all(not cell for row in self.__board for cell in row)

    def copy(self):
        return LightsOutPuzzle(self.get_board())

    def successors(self):
        for row in range(self.__rows):
            for col in range(self.__cols):
                new_puzzle = self.copy()
                new_puzzle.perform_move(row, col)
                yield (row, col), new_puzzle

    def find_solution(self):
        def board_to_tuple(board):
            return tuple(tuple(row) for row in board)

        start_state = board_to_tuple(self.__board)
        if self.is_solved():
            return []

        frontier = deque()
        frontier.append((start_state, []))
        in_frontier = {start_state}
        explored = set()

        while frontier:
            state, path = frontier.popleft()
            in_frontier.remove(state)
            puzzle = LightsOutPuzzle([list(row) for row in state])
            explored.add(state)

            for move, new_puzzle in puzzle.successors():
                new_state = board_to_tuple(new_puzzle.get_board())
                if new_state in explored or new_state in in_frontier:
                    continue
                if new_puzzle.is_solved():
                    return path + [move]
                frontier.append((new_state, path + [move]))
                in_frontier.add(new_state)

        return None


def create_puzzle(rows, cols):
    return LightsOutPuzzle([[False for _ in range(cols)] for _ in range(rows)])

############################################################
# Section 3: Linear Disk Movement
############################################################


def solve_identical_disks(length, n):
    start = tuple(range(n))
    goal = tuple(range(length - n, length))

    if start == goal:
        return []

    frontier = deque([(start, [])])
    explored = set()
    in_frontier = {start}

    while frontier:
        positions, path = frontier.popleft()
        in_frontier.remove(positions)
        if positions in explored:
            continue
        explored.add(positions)
        occupied = set(positions)

        for from_cell in positions:
            for delta in (-1, 1, -2, 2):
                to_cell = from_cell + delta
                if not (0 <= to_cell < length):
                    continue
                if to_cell in occupied:
                    continue
                if abs(delta) == 2:
                    mid = (from_cell + to_cell) // 2
                    if mid not in occupied:
                        continue
                next_positions = sorted(
                    (to_cell if pos == from_cell else pos for pos in positions)
                )
                next_state = tuple(next_positions)

                if next_state in explored or next_state in in_frontier:
                    continue
                next_path = path + [(from_cell, to_cell)]
                if next_state == goal:
                    return next_path
                frontier.append((next_state, next_path))
                in_frontier.add(next_state)

    return None


def solve_distinct_disks(length, n):
    start = tuple(list(range(n)) + [-1] * (length - n))
    goal = tuple([-1] * (length - n) + list(range(n - 1, -1, -1)))

    if start == goal:
        return []

    frontier = deque([(start, [])])
    explored = set()
    in_frontier = {start}
    empty = -1

    while frontier:
        board, path = frontier.popleft()
        in_frontier.remove(board)
        if board in explored:
            continue
        explored.add(board)

        for from_cell, disk in enumerate(board):
            if disk == empty:
                continue
            for delta in (-1, 1, -2, 2):
                to_cell = from_cell + delta
                if not (0 <= to_cell < length):
                    continue
                if board[to_cell] != empty:
                    continue
                if abs(delta) == 2:
                    mid = (from_cell + to_cell) // 2
                    if board[mid] == empty:
                        continue

                next_board = list(board)
                next_board[to_cell] = disk
                next_board[from_cell] = empty
                next_state = tuple(next_board)

                if next_state in explored or next_state in in_frontier:
                    continue
                next_path = path + [(from_cell, to_cell)]
                if next_state == goal:
                    return next_path
                frontier.append((next_state, next_path))
                in_frontier.add(next_state)

    return None

############################################################
# Section 4: Feedback
############################################################


# Just an approximation is fine.
feedback_question_1 = """
3H
"""

feedback_question_2 = """
Section2 Find Solution
"""

feedback_question_3 = """
What I like: I can see how both DFS and BFS work for same question.
Improvement: Give some cases
"""
