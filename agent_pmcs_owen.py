import math
import numpy as np
from scipy.signal import convolve2d
import random

ROW, COL = 6, 7

class PMCSAgent:
    def __init__(self, inc = 1000):
        horizontal_kernel = np.array([[1, 1, 1, 1]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(4, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)
        self.detection_kernels = [horizontal_kernel,
                                vertical_kernel, diag1_kernel, diag2_kernel]
        self.num_discs = -1
        self.inc = inc
        

    def result(self, s, player, a, inplace = False):
        """return state based on given action"""
        if inplace:
            s_cpy = s
        else:
            s_cpy = s.copy()
        for row in range(ROW-1, -1, -1):
            if s_cpy[row, a] == 0:
                s_cpy[row, a] = player
                break

        return s_cpy

    def check_win(self, board):
        # 2 not in terminal state, 0 draw
        for kernel in self.detection_kernels:
            a = convolve2d(board, kernel, mode='valid')
            if ((a == 4).any()):
                return 1
            if ((a == -4).any()):
                return -1

        # check if not in terminal state
        if 0 in board:
            return 2
        
        return 0

    def utility(self, s, player=1):
        """check is a state is terminal and return the utility if it is. None means not a terminal mode."""
        goal = self.check_win(s)        
        if goal == player: return 1
        if goal == 0: return 0  
        if goal == -1 * player: return -1  # loss is failure
        return None # continue

    def actions(self, s):
        """return non-full column index with ordering """
        res = []
        for col in range(COL):
            for row in range(ROW):
                if s[row, col] == 0:
                    res.append(col)
                    break

        return res

    def playout(self, state, action, player = 1):
        """Perfrom a random playout starting with the given action on the given board 
        and return the utility of the finished game."""
        state = state.copy()
        state = self.result(state, player, action, inplace=True)
        current_player = -1 * player
        num_discs = self.num_discs + 1
        while(True):
            # reached terminal state?
            if num_discs >= 7:
                u = self.utility(state, player)
                if u is not None: return u
            
            a = random.choice(self.actions(state))
            state = self.result(state, current_player, a, inplace=True)
            num_discs += 1
            
            # switch between players
            current_player *= -1

    def playouts(self, board, action, player, N):
        """Perform N playouts following the given action for the given board."""
        return [self.playout(board, action, player) for _ in range(N) ]


    def act(self, board, player = 1):
        """Pure Monte Carlo Search. Returns the action that has the largest average utility.
        The N playouts are evenly divided between the possible actions."""
        
        self.num_discs = len(np.where(board != 0)[0])
        N = (self.num_discs+1) * self.inc
        
        acts = self.actions(board)
        n = math.floor(N/COL)

        ps = {i : np.mean(self.playouts(board, i, player, N = n)) for i in acts }
        action = max(ps, key=ps.get)
        return action

if __name__ == "__main__":
    import time
    a = PMCSAgent(15000)
    board = np.zeros((ROW, COL), dtype=np.int8)
    st = time.time()
    a.act(board, 1)
    ut = time.time() - st
    print(ut)
