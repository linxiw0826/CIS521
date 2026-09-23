############################################################
# CIS 521: Homework 4
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
# Section 1: Dominoes Game
############################################################


def create_dominoes_game(rows, cols):
    return DominoesGame([[False for _ in range(cols)] for _ in range(rows)])


class DominoesGame(object):

    # Required
    def __init__(self, board):
        self.board = copy.deepcopy(board)
        self.initial_board = copy.deepcopy(board)

    def get_board(self):
        return self.board

    def reset(self):
        self.board = copy.deepcopy(self.initial_board)

    def is_legal_move(self, row, col, vertical):
        rows = len(self.board)
        cols = len(self.board[0]) if rows else 0
        if not (0 <= row < rows and 0 <= col < cols):
            return False
        if vertical:
            if row + 1 >= rows:
                return False
            return not self.board[row][col] and not self.board[row + 1][col]
        if col + 1 >= cols:
            return False
        return not self.board[row][col] and not self.board[row][col + 1]

    def legal_moves(self, vertical):
        for row in range(len(self.board)):
            for col in range(len(self.board[0]) if self.board else 0):
                if self.is_legal_move(row, col, vertical):
                    yield (row, col)

    def perform_move(self, row, col, vertical):
        if vertical:
            self.board[row][col] = True
            self.board[row + 1][col] = True
        else:
            self.board[row][col] = True
            self.board[row][col + 1] = True

    def game_over(self, vertical):
        return not any(self.legal_moves(vertical))

    def copy(self):
        return DominoesGame(self.board)

    def successors(self, vertical):
        for move in self.legal_moves(vertical):
            successor = self.copy()
            successor.perform_move(move[0], move[1], vertical)
            yield move, successor

    def get_random_move(self, vertical):
        return random.choice(list(self.legal_moves(vertical)))

    # Required
    def get_best_move(self, vertical, limit):
        # The utility is the number of available moves for the player whose
        # turn it is minus the number available to the other player.
        leaves = [0]

        def search(game, turn, depth, alpha, beta):
            moves = list(game.legal_moves(turn))
            if depth == 0 or not moves:
                leaves[0] += 1
                return (len(list(game.legal_moves(vertical))) -
                        len(list(game.legal_moves(not vertical))))

            maximizing = turn == vertical
            if maximizing:
                value = -math.inf
                for move in moves:
                    child = game.copy()
                    child.perform_move(move[0], move[1], turn)
                    value = max(value, search(child, not turn, depth - 1,
                                              alpha, beta))
                    alpha = max(alpha, value)
                    if alpha >= beta:
                        break
                return value

            value = math.inf
            for move in moves:
                child = game.copy()
                child.perform_move(move[0], move[1], turn)
                value = min(value, search(child, not turn, depth - 1,
                                          alpha, beta))
                beta = min(beta, value)
                if alpha >= beta:
                    break
            return value

        moves = list(self.legal_moves(vertical))
        if not moves:
            return None, 0, 1

        best_move = moves[0]
        best_value = -math.inf
        alpha = -math.inf
        beta = math.inf
        for move in moves:
            child = self.copy()
            child.perform_move(move[0], move[1], vertical)
            value = search(child, not vertical, max(0, limit - 1),
                           alpha, beta)
            if value > best_value:
                best_value = value
                best_move = move
            alpha = max(alpha, best_value)

        return best_move, best_value, leaves[0]

############################################################
# Section 2: Feedback
############################################################


# Just an approximation is fine.
feedback_question_1 = """
Implementing the minimax search with alpha-beta pruning.
"""

feedback_question_2 = """
How game-playing algorithms represent the state of a game and search through
possible future moves.
"""

feedback_question_3 = """
I would write more small test cases.
"""
