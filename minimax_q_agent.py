import numpy as np
from hyperparameters import *
from soccer_env import *
from scipy.optimize import linprog
import pandas as pd

def solveLp_row(a):
    newA = -1 * np.transpose(a)
    f = 1 * np.ones(newA.shape[1])
    b = -1 * np.ones(newA.shape[0])
    result = linprog(f, newA, b)
    x = result['x'] / (1 * result['fun'])
    v = np.dot(np.transpose(a), x)
    return x, v.min()


class MiniMax_Q_Agent:
    def __init__(self, id, env, alpha=LEARNING_RATE, decay=DECAY, gamma=DISCOUNT_FACTOR, epsilon=EXPLORE_EPSILON,
                 lr=LEARNING_RATE, training=True):
        self.id = id
        self.V = np.ones(env.state_space_size)
        # state action value table, num_state*num_action_p1*num_action_p2
        self.Q = np.ones((env.state_space_size, env.action_space_size, env.action_space_size))
        self.PI = np.ones((env.state_space_size, env.action_space_size)) / env.action_space_size
        self.lr = lr
        self.epsilon = epsilon
        self.alpha = alpha
        self.decay = decay
        self.gamma = gamma
        self.state_num = env.state_space_size
        self.action_num = env.action_space_size
        self.training = training
        self.env = SoccerEnv()

    def take_action(self, state):
        # check your note!!
        if self.training and np.random.rand() < self.epsilon:
            action = np.random.randint(0, self.action_num)
        else:
            action = np.random.choice([i for i in range(self.action_num)], size=None, p=self.PI[state])
            # action = np.argmax(self.PI[state])
        return action

    def update_Qval(self, a, o, s, next_s,reward):
        self.Q[s, a, o] = (1 - self.alpha) * self.Q[s, a, o] + self.alpha * (reward + self.gamma * self.V[next_s])

    def update_PI(self, state):
        c = np.zeros(self.action_num+1)
        c[0] =-1
        A_ub = np.ones((self.action_num,self.action_num+1))
        A_ub[:,1:] = -self.Q[state].T
        b_ub = np.zeros(self.action_num)
        A_eq = np.ones((1,self.action_num+1))
        A_eq[0,0] = 0
        b_eq = [1]
        bounds = ((None, None),) + ((0, 1),) * self.action_num
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        for i in range(1,len(res.x)):
            if res.x[i]<0:
                res.x[i] = 0
        if res.success:
            self.PI[state] = res.x[1:]
            self.V[state] = res.x[0]

    def save_agent(self,path = MODEL_PATH):
        PI = pd.DataFrame(self.PI)
        # Q = pd.DataFrame(self.Q)
        V = pd.DataFrame(self.V)
        PI.to_csv(path + 'PI.txt', sep=' ', header=False, index=False)
        # Q.to_csv(path + 'Q.txt', sep=' ', header=False, index=False)
        V.to_csv(path + 'V.txt', sep=' ', header=False, index=False)

    def load_agent(self,path = MODEL_PATH):
        PI = pd.read_csv(path + 'PI.txt', sep=' ', header=False, index=False)
        # Q = pd.read_csv(path + 'Q.txt', sep=' ', header=False, index=False)
        V = pd.read_csv(path + 'V.txt', sep=' ', header=False, index=False)
        self.PI = PI.values
        # self.Q = Q.values
        self.V = V.values