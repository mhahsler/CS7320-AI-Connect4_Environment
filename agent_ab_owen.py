import math
from copy import deepcopy

def result(s, player, a):
    """return state based on given action"""
    s_cpy = deepcopy(s)
    NUM_ROW = len(s)
    for row in range(NUM_ROW-1, -1, -1):
        if s_cpy[row][a] == 0:
            s_cpy[row][a] = player
            break
            
    return s_cpy

def check_win(board):
    """check the board and return one of 1, -1, d (draw), or n (for next move)"""
    
    ROW, COL = len(board), len(board[0])
    
    # columns 
    for col in range(COL):
        cnt = 1
        for row in range(ROW-2, -1, -1):
            if board[row][col] == 0: break
            if board[row][col] == board[row+1][col]:
                cnt += 1
            else:
                cnt = 1
            if cnt == 4:
                return board[row][col]
    
    # rows
    for row in range(ROW):
        cnt = 1
        for col in range(COL-1):
            if board[row][col] == board[row][col+1]:
                cnt += 1
            else:
                cnt = 1
            if board[row][col] != 0 and cnt == 4:
                return board[row][col]
    
    # diagonals
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
                if board[row][col] != 0 and cnt == 4:
                    return board[row][col]
                row -= 1
                col -= 1
                
    # diagonals in another direction
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
                if board[row][col] != 0 and cnt == 4:
                    return board[row][col]
                row -= 1
                col += 1

    # check if not in terminal state
    for row in range(ROW):
        for col in range(COL):
            if board[row][col] == 0:
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
        ROW, COL = len(s), len(s[0])
        res = []
        priority = [0 if i % 2 == 0 else 1  for i in range(COL)]
        for col in range(COL):
            for row in range(ROW):
                if s[row][col] == 0:
                    res.append((priority[col],col))
                    break
                
        actions = [a for _, a in sorted(res, reverse=True)]
        
        return actions



# Your code/ answer goes here.
DEBUG = 0 # 1 ... count nodes, 2 ... debug each node
COUNT = 0

def alpha_beta_search(board, player = 1):
    """start the search."""
    global DEBUG, COUNT
    COUNT = 0
    
    value, move = max_value_ab(board, player, -math.inf, +math.inf)
    
    if DEBUG >= 1: print(f"Number of nodes searched: {COUNT}") 
    
    return { "move": move, "value": value }

def max_value_ab(state, player, alpha, beta):
    """player's best move."""
    global DEBUG, COUNT
    COUNT += 1
       
    # return utility of state is a terminal state
    v = utility(state, player)
    if DEBUG >= 2: print(f"max: {state} [alpha,beta]=[{alpha},{beta}] v={v}")
    if v is not None: 
        if DEBUG >= 2: print(f"     found terminal state. backtracking.")
        return v, None
        
    v, move = -math.inf, None

    # check all possible actions in the state, update alpha and return move with the largest value
    for a in actions(state):
        v2, a2 = min_value_ab(result(state, player, a), player, alpha, beta)
        if DEBUG >= 2: print(f"max: {state} (backtracked) [alpha,beta]=[{alpha},{beta}] v={v2}")
        
        if v2 > v:
            v, move = v2, a
            alpha = max(alpha, v)
        if v >= beta:
            if DEBUG >= 2: print(f"     v>=beta ({v}>={beta}): pruning remaining subtree (actions). backtracking.")
            return v, move
    
    return v, move

def min_value_ab(state, player, alpha, beta):
    """opponent's best response."""
    global DEBUG, COUNT
    COUNT += 1
    
    # return utility of state is a terminal state
    v = utility(state, player)
    if DEBUG >= 2: print(f"min: {state} [alpha,beta]=[{alpha},{beta}] v={v}")
    if v is not None: 
        if DEBUG >= 2: print(f"     found terminal state. backtacking.")
        return v, None
    
    v, move = +math.inf, None

    # check all possible actions in the state, update beta and return move with the smallest value
    for a in actions(state):
        v2, a2 = max_value_ab(result(state, -1 * player, a), player, alpha, beta)
        if DEBUG >= 2: print(f"min: {state} (backtracked) [alpha,beta]=[{alpha},{beta}] v={v2}")
        
        if v2 < v:
            v, move = v2, a
            beta = min(beta, v)
        if v <= alpha: 
            if DEBUG >= 2: print(f"     v<=alpha ({v}<={alpha}): pruning remaining subtree (actions). backtracking.")
            return v, move
    
    return v, move

def alpha_beta_agent(board, player):
    return alpha_beta_search(board, player)['move']

class AlphaBetaAgent:
    def __init__(self, name = "AlphaBetaAgent"):
        self.name = name
        self.state = None
    
    def act(self, board, player):
        move = alpha_beta_search(board, player)['move']
        self.state = result(board, player, move)
        return move
    
