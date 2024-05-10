import math
import numpy as np
from scipy.signal import convolve2d
import random

class FastPMCSAgent:
    def __init__(self, inc = 1000, check_opponent_win=True, huristic = True, clear_cache=False):
        horizontal_kernel = np.array([[1, 1, 1, 1]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(4, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)
        self.detection_kernels = [horizontal_kernel, vertical_kernel, 
                                  diag1_kernel, diag2_kernel]
        
        self.inc = inc
        self.hurisitc = huristic
        self.ROW, self.COL = 6, 7
        self.cache = {}
        self.clear_cache=clear_cache
        self.check_opponent_win = check_opponent_win

    def playout(self, state, action, player = 1):
        """Perfrom a random playout starting with the given action on the given board 
        and return the utility of the finished game."""
        state = state.copy()
        action_dict = self.action_dict.copy()
        action_keys = self.action_keys.copy()

        num_discs = self.num_discs
        current_player = player
        while(True):
            state[action_dict[action]-1, action] = current_player
            num_discs += 1
            # reached terminal state?
            if num_discs >= 7:
                state_bytes = state.tobytes()
                if state_bytes in self.cache:
                    u = self.cache[state_bytes]
                else:
                    u = 0
                    for kernel in self.detection_kernels:
                        a = convolve2d(state, kernel, mode='valid')
                        if ((a == 4).any()):
                            u = player
                            break
                        if ((a == -4).any()):
                            u = -player
                            break
                    self.cache[state_bytes] = u

                if u != 0 or num_discs == 42:
                    if u == 1 and self.hurisitc:
                        u += self.hurisitc_score(state, player)
                    return u

            action_dict[action] -= 1
            if action_dict[action] == 0:
                action_keys.remove(action)
                
            action = random.choice(action_keys)
        
            # switch between players
            current_player = -current_player
    
    def playouts(self, board, action, player, N):
        """Perform N playouts following the given action for the given board."""
        return [self.playout(board, action, player) for _ in range(N)]


    def act(self, board, player = 1):
        """Pure Monte Carlo Search. Returns the action that has the largest average utility.
        The N playouts are evenly divided between the possible actions."""
        
        self.num_discs = len(np.where(board != 0)[0])
        if self.clear_cache and self.num_discs <= 1:
            self.cache = {} # reset at each game starts
    
        if self.num_discs == 0:
            return 3

        self.action_dict = {}
        for col in range(self.COL):
            num_zeros = len(np.where(board[:, col] == 0)[0])
            if num_zeros == 0:
                continue
            self.action_dict[col] = num_zeros
        self.action_keys = list(self.action_dict)
        
        if self.check_opponent_win:
            win_move = self.opponent_win_move(board, player)
            if win_move != -1:
                return win_move

        N = (self.num_discs+1) * self.inc
        n = math.floor(N/self.COL)
        
        ps = {i : np.mean(self.playouts(board, i, player, N = n)) for i in self.action_keys}
        action = max(ps, key=ps.get)
        return action
    
    def opponent_win_move(self, board, player):
        opponent_player = -player
        for a in self.action_keys:
            new_board = board.copy()
            new_board[self.action_dict[a]-1, a] = opponent_player

            for kernel in self.detection_kernels:
                convolved = convolve2d(new_board, kernel, mode='valid')
                if ((convolved == 4*opponent_player).any()):
                    return a
        return -1
    
    def hurisitc_score(self, board, player):
        score = 0
        for kernel in self.detection_kernels:
            a = convolve2d(board, kernel, mode='valid')
            score += np.sum(a == player*3) * 0.1
        if score > 0:
            score -= 0.1
        
        return score

if __name__ == "__main__":
    pass