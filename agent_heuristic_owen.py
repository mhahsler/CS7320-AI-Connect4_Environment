import math
from copy import deepcopy
import numpy as np
from scipy.signal import convolve2d

horizontal_kernel = np.array([[1, 1, 1, 1]])
vertical_kernel = np.transpose(horizontal_kernel)
diag1_kernel = np.eye(4, dtype=np.uint8)
diag2_kernel = np.fliplr(diag1_kernel)
detection_kernels = [horizontal_kernel,
                        vertical_kernel, diag1_kernel, diag2_kernel]
ROW, COL = 6, 7
score_one = 1/(ROW*COL)
priority = [0 if i % 2 == 0 else 1  for i in range(COL)]

def result(s, player, a):
    """return state based on given action"""
    s_cpy = s.copy()
    for row in range(ROW-1, -1, -1):
        if s_cpy[row, a] == 0:
            s_cpy[row, a] = player
            break

    return s_cpy

def check_win(board):
    for kernel in detection_kernels:
        a = convolve2d(board, kernel, mode='valid')
        if ((a == 4).any()):
            return 1
        if ((a == -4).any()):
            return -1

    # check if not in terminal state
    for row in range(ROW):
        for col in range(COL):
            if board[row, col] == 0:
                return 'n'
            
    return 'd'

def terminal(s):
    """check if s is terminal state"""
    return check_win(s) != 'n'

def utility(s, player=1):
   """check is a state is terminal and return the utility if it is. None means not a terminal mode."""
   goal = check_win(s)        
   if goal == player: return 1
   if goal == 'd': return 0  
   if goal == -1 * player: return -1  # loss is failure
   return None # continue

def actions(s, player = 1):
    """return non-full column index with ordering """
    res = []
    for col in range(COL):
        for row in range(ROW):
            if s[row, col] == 0:
                res.append((priority[col],col))
                break
            
    actions = [a for _, a in sorted(res, reverse=True)]
    
    return actions

def is_valid(board, row, col, check_below_4th):
    if row < 0 or row >= ROW or col < 0 or col >= COL\
        or board[row, col] != 0:
        return False
    if not check_below_4th:
        return True
    if row == ROW - 1:
        return True
    if board[row+1, col] == 0:
        return False
    return True
    
def eval_fun(board, player = 1, check_below_4th = True):
    """heuristic for utility of state. Returns score for a node:
    1. For terminal states it returns the utility. 
    2. For non-terminal states, it calculates a weighted linear function using features of the state. 
    The features we look at are 3 in a row/col/diagonal where the 4th square is empty. We assume that
    the more of these positions we have, the higher the chance of winning.
    We need to be careful that the utility of the heuristic stays between [-1,1]. 
    Note that the largest possible number of these positions is smaller than ROW*COL. I weigh the count by 1/(ROW*COL), 
    guaranteeing that is in the needed range.
    
    Function Returns: heuistic value, terminal?"""
    
    # terminal state?
    u = utility(board, player)
    if u is not None: return u, True
      
    score = 0
    
    
    # columns 
    for col in range(COL):
        cnt = 1
        for row in range(ROW-2, -1, -1):
            if board[row][col] == 0: break
            if board[row][col] == board[row+1][col]:
                cnt += 1
            else:
                cnt = 1
            if cnt == 3 and row-1>=0 and board[row-1][col] == 0:
                score += board[row][col] * score_one 
    
    # rows
    for row in range(ROW):
        cnt = 1
        for col in range(COL-1):
            if board[row][col] == board[row][col+1]:
                cnt += 1
            else:
                cnt = 1
            if board[row][col] != 0 and cnt == 3 and \
                (is_valid(board, row, col-2, check_below_4th) or is_valid(board, row, col+2, check_below_4th)):
                score += board[row][col] * score_one 
    
    # diagonals \ 
    for start_col in range(3, COL):
        if start_col != COL - 1:
            start_rows = (ROW - 1,)
        else:
            start_rows = range(ROW-1, 2, -1)
            
        for start_row in start_rows:
            row, col = start_row - 1, start_col-1
            cnt = 1
            while row >= 0 and col >= 0:
                if board[row][col] == board[row+1][col+1]:
                    cnt += 1
                else:
                    cnt = 1
                if board[row][col] != 0 and cnt == 3 and \
                    (is_valid(board, row+3, col+3, check_below_4th) or is_valid(board, row-1, col-1, check_below_4th)):
                    score += board[row][col] * score_one 
                row -= 1
                col -= 1
                
    # diagonals in another direction /
    for start_col in range(0, COL-4+1):
        if start_col != 0:
            start_rows = (ROW - 1,)
        else:
            start_rows = range(ROW-1, 2, -1)
            
        for start_row in start_rows:
            row, col = start_row - 1, start_col + 1
            cnt = 1
            while row >= 0 and col < COL:
                if board[row][col] == board[row+1][col-1]:
                    cnt += 1
                else:
                    cnt = 1
                if board[row][col] != 0 and cnt == 3 and \
                    (is_valid(board, row+3, col-3, check_below_4th) or is_valid(board, row-1, col+1, check_below_4th)):
                    score += board[row][col] * score_one 
                row -= 1
                col += 1
    
    return score, False

# global variables
DEBUG = 0 # 1 ... count nodes, 2 ... debug each node
COUNT = 0

def alpha_beta_search(board, cutoff = None, check_below_4th = True, player = 1):
    """start the search. cutoff = None is minimax search with alpha-beta pruning."""
    global DEBUG, COUNT
    COUNT = 0

    value, move = max_value_ab(board, player, -math.inf, +math.inf, 0, cutoff, check_below_4th)
    
    if DEBUG >= 1: print(f"Number of nodes searched (cutoff = {cutoff}): {COUNT}") 
    
    return {"move": move, "value": value}

def max_value_ab(state, player, alpha, beta, depth, cutoff, check_below_4th = True):
    """player's best move."""
    global DEBUG, COUNT
    COUNT += 1
    
    # cut off and terminal test
    v, terminal = eval_fun(state, player, check_below_4th)
    if((cutoff is not None and depth >= cutoff) or terminal): 
        if(terminal): 
            alpha, beta = v, v
        if DEBUG >= 2: print(f"stopped at {depth}: {state} term: {terminal} eval: {v} [{alpha}, {beta}]" ) 
        return v, None
    
    v, move = -math.inf, None

    # check all possible actions in the state, update alpha and return move with the largest value
    for a in actions(state):
        v2, a2 = min_value_ab(result(state, player, a), player, alpha, beta, depth + 1, cutoff, check_below_4th)
        if v2 > v:
            v, move = v2, a
            alpha = max(alpha, v)
        if v >= beta: return v, move
    
    return v, move

def min_value_ab(state, player, alpha, beta, depth, cutoff, check_below_4th = True):
    """opponent's best response."""
    global DEBUG, COUNT
    COUNT += 1
    
    # cut off and terminal test
    v, terminal = eval_fun(state, player, check_below_4th)
    if((cutoff is not None and depth >= cutoff) or terminal): 
        if(terminal): 
            alpha, beta = v, v
        if DEBUG >= 2: print(f"stopped at {depth}: {state} term: {terminal} eval: {v} [{alpha}, {beta}]" ) 
        return v, None
    
    v, move = +math.inf, None

    # check all possible actions in the state, update beta and return move with the smallest value
    for a in actions(state):
        v2, a2 = max_value_ab(result(state, -1 * player, a), player, alpha, beta, depth + 1, cutoff, check_below_4th)
        if v2 < v:
            v, move = v2, a
            beta = min(beta, v)
        if v <= alpha: return v, move
    
    return v, move


def heurisitic_agent(board, player, cutoff = 8, check_below_4th = True):
    return alpha_beta_search(board, cutoff, check_below_4th, player)['move']

class HeuristicAgent:
    def __init__(self, cutoff = 8, check_below_4th = True, name = "HeuristicAgent"):
        self.name = name
        self.state = None
        self.cutoff = cutoff
        self.check_below_4th = check_below_4th
    
    def act(self, board, player):
        move = alpha_beta_search(board, self.cutoff , player)['move']
        self.state = result(board, player, move)
        return move