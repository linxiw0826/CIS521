############################################################
# CIS 521: Homework 3
############################################################

############################################################
# Imports
############################################################

# Include your imports here, if any are used.
from collections import deque
import heapq
import math
import random

############################################################

student_name = "Linxi Wu"

############################################################
# Section 1: Tile Puzzle
############################################################


def create_tile_puzzle(rows, cols):
    board = [[r * cols + c + 1 for c in range(cols)] for r in range(rows)]
    board[-1][-1] = 0
    return TilePuzzle(board)


class TilePuzzle(object):

    # Required
    def __init__(self, board):
        self.board = [row[:] for row in board]
        self.rows = len(self.board)
        self.cols = len(self.board[0])

    def get_board(self):
        return [row[:] for row in self.board]

    def perform_move(self, direction):
        offsets = {"up": (-1, 0), "down": (1, 0),
                   "left": (0, -1), "right": (0, 1)}
        if direction not in offsets:
            return False
        r, c = next((r, c) for r in range(self.rows)
                    for c in range(self.cols) if self.board[r][c] == 0)
        dr, dc = offsets[direction]
        nr, nc = r + dr, c + dc
        if not (0 <= nr < self.rows and 0 <= nc < self.cols):
            return False
        self.board[r][c], self.board[nr][nc] = (
            self.board[nr][nc], self.board[r][c])
        return True

    def scramble(self, num_moves):
        for _ in range(num_moves):
            self.perform_move(random.choice(("up", "down", "left", "right")))

    def is_solved(self):
        goal = list(range(1, self.rows * self.cols)) + [0]
        return sum(self.board, []) == goal

    def copy(self):
        return TilePuzzle(self.board)

    def successors(self):
        for direction in ("up", "down", "left", "right"):
            child = self.copy()
            if child.perform_move(direction):
                yield direction, child

    # Required
    def find_solutions_iddfs(self):
        def search(puzzle, depth, path, visited):
            if puzzle.is_solved():
                yield path
            elif depth:
                for move, child in puzzle.successors():
                    state = tuple(sum(child.board, []))
                    if state not in visited:
                        yield from search(child, depth - 1, path + [move],
                                          visited | {state})
        depth = 0
        while True:
            solutions = list(search(self, depth, [],
                                    {tuple(sum(self.board, []))}))
            if solutions:
                for solution in solutions:
                    yield solution
                return
            depth += 1

    # Required
    def find_solution_a_star(self):
        def h(puzzle):
            total = 0
            for r in range(puzzle.rows):
                for c in range(puzzle.cols):
                    value = puzzle.board[r][c]
                    if value:
                        gr, gc = divmod(value - 1, puzzle.cols)
                        total += abs(r - gr) + abs(c - gc)
            return total
        start = tuple(sum(self.board, []))
        heap = [(h(self), 0, 0, start, self, [])]
        best = {start: 0}
        counter = 1
        while heap:
            _, cost, _, state, puzzle, path = heapq.heappop(heap)
            if puzzle.is_solved():
                return path
            if cost != best[state]:
                continue
            for move, child in puzzle.successors():
                child_state = tuple(sum(child.board, []))
                new_cost = cost + 1
                if new_cost < best.get(child_state, float("inf")):
                    best[child_state] = new_cost
                    counter += 1
                    heapq.heappush(heap, (new_cost + h(child), new_cost,
                                          counter, child_state, child,
                                          path + [move]))
        return None

############################################################
# Section 2: Grid Navigation
############################################################


def find_path(start, goal, scene):
    rows, cols = len(scene), len(scene[0])
    if (not (0 <= start[0] < rows and 0 <= start[1] < cols) or
            not (0 <= goal[0] < rows and 0 <= goal[1] < cols) or
            scene[start[0]][start[1]] or scene[goal[0]][goal[1]]):
        return None

    def distance(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    directions = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
                  (0, 1), (1, -1), (1, 0), (1, 1))
    heap = [(distance(start, goal), 0, 0, start)]
    parent = {start: None}
    best = {start: 0}
    counter = 1
    while heap:
        _, cost, _, current = heapq.heappop(heap)
        if cost != best[current]:
            continue
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]
        for dr, dc in directions:
            nxt = (current[0] + dr, current[1] + dc)
            if not (0 <= nxt[0] < rows and 0 <= nxt[1] < cols):
                continue
            if scene[nxt[0]][nxt[1]]:
                continue
            new_cost = cost + math.sqrt(dr * dr + dc * dc)
            if new_cost < best.get(nxt, float("inf")):
                best[nxt] = new_cost
                parent[nxt] = current
                counter += 1
                priority = new_cost + distance(nxt, goal)
                heapq.heappush(heap, (priority, new_cost, counter, nxt))
    return None

############################################################
# Section 3: Linear Disk Movement, Revisited
############################################################


def solve_distinct_disks(length, n):
    start = tuple(range(n)) + (None,) * (length - n)
    goal = (None,) * (length - n) + tuple(range(n - 1, -1, -1))
    goal_positions = {disk: length - 1 - disk for disk in range(n)}

    def heuristic(state):
        total = 0
        for i, disk in enumerate(state):
            if disk is not None:
                total += (abs(goal_positions[disk] - i) + 1) // 2
        return total

    heap = [(heuristic(start), 0, 0, start)]
    parent = {start: None}
    moves = {}
    best = {start: 0}
    counter = 1
    while heap:
        _, cost, _, state = heapq.heappop(heap)
        if cost != best[state]:
            continue
        if state == goal:
            answer = []
            while parent[state] is not None:
                answer.append(moves[state])
                state = parent[state]
            return answer[::-1]
        for i, disk in enumerate(state):
            if disk is None:
                continue
            for step in (-2, -1, 1, 2):
                j = i + step
                if not (0 <= j < length) or state[j] is not None:
                    continue
                if abs(step) == 2 and state[i + step // 2] is None:
                    continue
                child = list(state)
                child[i], child[j] = child[j], child[i]
                child = tuple(child)
                new_cost = cost + 1
                if new_cost < best.get(child, float("inf")):
                    best[child] = new_cost
                    parent[child] = state
                    moves[child] = (i, j)
                    counter += 1
                    priority = new_cost + heuristic(child)
                    heapq.heappush(heap, (priority, new_cost,
                                          counter, child))
    return None

############################################################
# Section 4: Feedback
############################################################


# Just an approximation is fine.
feedback_question_1 = """
The most interesting part is IDDFS of Tile Puzzle.
"""

feedback_question_2 = """
I think it is harder than previous two homeworks.
"""

feedback_question_3 = """
I have better understanding of BFS, IDDFS, and A*.
"""
