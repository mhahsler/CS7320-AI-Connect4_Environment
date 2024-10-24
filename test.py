import numpy as np
import time

N_ROWS = 6
N_COLS = 7
WINNING_LENGTH = 4

# Transposition table to cache minimax results
transposition_table = {}

def minimax_agent1(board, player, max_depth=5, timeout=5):
    actions = available_actions(board)
    best_action = None
    best_score = -np.inf
    start_time = time.time()

    # Iterative deepening loop
    for depth in range(1, max_depth + 1):
        for action in actions:
            new_board = place(board.copy(), action, player)
            score = alpha_beta_minimax(new_board, depth, -np.inf, np.inf, False, player, start_time, timeout)
           
            if score > best_score:
                best_score = score
                best_action = action

        # Check if time limit exceeded
        if time.time() - start_time > timeout:
            break

    return best_action

def alpha_beta_minimax(board, depth, alpha, beta, maximizing_player, player, start_time, timeout):
    # Check transposition table for cached result
    board_key = tuple(board.flatten())
    if board_key in transposition_table:
        return transposition_table[board_key]

    if depth == 0 or check_win(board) is not None or time.time() - start_time > timeout:
        eval_score = evaluate(board, player)
        transposition_table[board_key] = eval_score
        return eval_score

    actions = available_actions(board)

    if maximizing_player:
        max_eval = -np.inf
        for action in actions:
            new_board = place(board.copy(), action, player)
            eval = alpha_beta_minimax(new_board, depth - 1, alpha, beta, False, player, start_time, timeout)
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        transposition_table[board_key] = max_eval
        return max_eval
    else:
        min_eval = np.inf
        for action in actions:
            new_board = place(board.copy(), action, -player)
            eval = alpha_beta_minimax(new_board, depth - 1, alpha, beta, True, player, start_time, timeout)
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        transposition_table[board_key] = min_eval
        return min_eval

def evaluate(board, player):
    score = 0
    score += evaluate_window(board, player)
    score -= evaluate_window(board, -player)
    return score

def evaluate_window(board, player):
    score = 0

    # Evaluate horizontal windows
    for r in range(N_ROWS):
        for c in range(N_COLS - WINNING_LENGTH + 1):
            window = board[r, c:c + WINNING_LENGTH]
            score += score_window(window, player)

    # Evaluate vertical windows
    for r in range(N_ROWS - WINNING_LENGTH + 1):
        for c in range(N_COLS):
            window = board[r:r + WINNING_LENGTH, c]
            score += score_window(window, player)

    # Evaluate positive slope diagonal windows
    for r in range(N_ROWS - WINNING_LENGTH + 1):
        for c in range(N_COLS - WINNING_LENGTH + 1):
            window = [board[r + i, c + i] for i in range(WINNING_LENGTH)]
            score += score_window(window, player)

    # Evaluate negative slope diagonal windows
    for r in range(N_ROWS - WINNING_LENGTH + 1):
        for c in range(WINNING_LENGTH - 1, N_COLS):
            window = [board[r + i, c - i] for i in range(WINNING_LENGTH)]
            score += score_window(window, player)

    return score

def score_window(window, player):
    opponent = -player
    player_count = np.sum(window == player)
    opponent_count = np.sum(window == opponent)
    empty_count = np.sum(window == 0)

    if opponent_count == 3 and empty_count == 1:
        return -500  # Block opponent's winning move
    elif player_count == 3 and empty_count == 1:
        return 100  # Favorable move
    elif player_count == 2 and empty_count == 2:
        return 10  # Another favorable configuration
    elif player_count == 1 and empty_count == 3:
        return 1  # Single disc in a row
    elif player_count == 4:
        return 1000  # Winning move
    else:
        return 0  # No significant advantage or disadvantage

def place(board, choice, player):
    board = board.copy()
    row = np.argmax(board[:, choice] == 0)
    board[row, choice] = player
    return board

def available_actions(board):
    return np.where(board[0] == 0)[0]

def check_win(board):
    # Check for a win in all directions
    rows = board.shape[0]
    cols = board.shape[1]

    # Check horizontal
    for r in range(rows):
        for c in range(cols - WINNING_LENGTH + 1):
            if all(board[r, c + i] == board[r, c] for i in range(WINNING_LENGTH)):
                return board[r, c]

    # Check vertical
    for r in range(rows - WINNING_LENGTH + 1):
        for c in range(cols):
            if all(board[r + i, c] == board[r, c] for i in range(WINNING_LENGTH)):
                return board[r, c]

    # Check positive slope diagonal
    for r in range(rows - WINNING_LENGTH + 1):
        for c in range(cols - WINNING_LENGTH + 1):
            if all(board[r + i, c + i] == board[r, c] for i in range(WINNING_LENGTH)):
                return board[r, c]

    # Check negative slope diagonal
    for r in range(WINNING_LENGTH - 1, rows):
        for c in range(cols - WINNING_LENGTH + 1):
            if all(board[r - i, c + i] == board[r, c] for i in range(WINNING_LENGTH)):
                return board[r, c]

    return None