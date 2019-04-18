#Grant Zhao
#260563769
import numpy as np
from scipy.optimize import linprog

part_a = np.array([[3, -1, 2], [1, 2, -2]], dtype=float)
part_b = np.array([[6, 0, 5, 6], [-3, 3, -4, 3], [8, 1, 2, 2]])
part_c = np.array([[6, 0, 5, 6], [-3, 3, -4, 3], [8, 1, 2, 2]])
def solveLp_column(a):
    f = -1 * np.ones(a.shape[1])
    b = np.ones(a.shape[0])
    result = linprog(f, A_ub=a, b_ub=b)
    x = result['x'] / (-1 * result['fun'])
    v = np.dot(a, x)
    return x, v.max()


def solveLp_row(a):
    newA = -1 * np.transpose(a)
    f = 1 * np.ones(newA.shape[1])
    b = -1 * np.ones(newA.shape[0])
    result = linprog(f, newA, b)
    x = result['x'] / (1 * result['fun'])
    v = np.dot(np.transpose(a), x)
    return x, v.min()


def find_strategies_and_game_val(game):
    x, v = solveLp_column(game)
    x1, v1 = solveLp_row(game)
    print('column player dominate strategy is ' + str(x) + ', the value of the game is ' + str(v))
    print('row player dominate strategy is ' + str(x1) + ', the value of the game is ' + str(v1))


find_strategies_and_game_val(part_b)
