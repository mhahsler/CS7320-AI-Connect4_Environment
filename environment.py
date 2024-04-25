
import math
import matplotlib.pyplot as plt
import numpy as np

from IPython.display import clear_output

import time
from timeit import default_timer as timer

from scipy.signal import convolve2d

def visualize(board):
    plt.axes()
    rectangle = plt.Rectangle(
        (-0.5, len(board)*-1+0.5), len(board[0]), len(board), fc='blue')
    circles = []
    for i, row in enumerate(board):
        for j, val in enumerate(row):
            color = 'white' if val == 0 else 'red' if val == 1 else 'yellow'
            circles.append(plt.Circle((j, i*-1), 0.4, fc=color))

    plt.gca().add_patch(rectangle)
    for circle in circles:
        plt.gca().add_patch(circle)

    plt.axis('scaled')
    plt.show()


class HelperFunctions:
    @staticmethod
    def place(choice: int, board, player: int):
        board = board.copy()
        
        if (board[0, choice] != 0):
            print(f"invalid Move! Player {player} loses its turn!")
            return board
        
        row = board.shape[0] - 1
        while (board[row, choice] != 0):
            row -= 1
        
        board[row, choice] = player
        return board

    @staticmethod
    def get_valid_moves(board):
        return np.where(board[0]==0)[0].tolist()

    @staticmethod
    def check_win(board):
        # check for win/loss
        horizontal_kernel = np.array([[1, 1, 1, 1]])
        vertical_kernel = np.transpose(horizontal_kernel)
        diag1_kernel = np.eye(4, dtype=np.uint8)
        diag2_kernel = np.fliplr(diag1_kernel)
        detection_kernels = [horizontal_kernel,
                             vertical_kernel, diag1_kernel, diag2_kernel]
        for kernel in detection_kernels:

            a = convolve2d(board, kernel, mode='valid')
            if ((a == 4).any()):
                return 1
            if ((a == -4).any()):
                return -1

        # check for draw
        if (len(HelperFunctions.get_valid_moves(board)) == 0):
            return 0

        return None

    @staticmethod
    def time_function(theFunc, *args):
        start = timer()
        theFunc(*args)
        print(f"{theFunc.__name__}: {(timer() - start) * 1000} ms")

    @staticmethod
    def calc_utility(player, board):
        winner = HelperFunctions.check_win(board)
        
        if (winner == None):
            raise Exception("Tried to calculate the utility of non-terminal state")

        if winner == 0:
            return 0
        if winner == player:
            return 1
        else:
            return -1


def empty_board(shape=(6, 7)):
    return np.full(shape=shape, fill_value=0)


def truly_dynamic_environment(players, size=(6, 7), visual=False, board=None):
    result = {}
    if board is None:
        board = empty_board(shape=size)
    turn_num = 0
    result['algo_info'] = {
        players[0]['name']: {'time': []},
        players[1]['name']: {'time': []}
    }
    result['algo_info']
    past_boards = []
    # While there is not a winner yet (0 does not mean draw in this case, it means non terminal state)
    while (HelperFunctions.check_win(board) == None):
        player_turn = turn_num % 2

        start = timer()
        
        # pass a copy of the board so the agent cannot cheat by changing the board
        choice = players[player_turn]['algo'](
            board.copy(), players[player_turn]['player'], **players[player_turn]['args'])
        end = timer()
        
        board = HelperFunctions.place(
            choice, board, player=players[player_turn]['player'])
        if visual:
            visualize(board)
            clear_output(wait=True)
        result['algo_info'][players[player_turn]['name']
                            ]['time'].append((end - start) * 1000)
        past_boards.append(board)
        turn_num += 1
    result['winner'] = HelperFunctions.check_win(board)
    result['turns_taken'] = turn_num
    for name in result['algo_info']:
        print(
            f"{name} took a total of {round(np.sum(result['algo_info'][name]['time'])/ 1000,3)} seconds")

    print(
        f"The winner is {players[(result['winner']-1)//-2]['name']} ({result['winner']})")
    print(f"Turns Taken: {turn_num}")

    return result, board, past_boards


def replay(all_boards, sleep_time: int = 1):
    for board in all_boards:
        visualize(board)
        time.sleep(sleep_time)
        clear_output(wait=True)


# print(__name__)
if __name__ == "__main__":

    board = [
        [-1,  0, -1, -1,  0,  0,  0],
        [1,  0, -1,  1,  0,  0,  0,],
        [1,  0, -1, -1,  1,  0,  0,],
        [1,  0,  1,  1, -1,  0,  0],
        [-1,  0, -1,  1,  1, 0,  0],
        [-1,  1,  1,  1, -1,  0,  0]

    ]

    visualize(board)
