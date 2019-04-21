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


class Q_Agent:
    def __init__(self, id, env, alpha=LEARNING_RATE, decay=DECAY, gamma=DISCOUNT_FACTOR, epsilon=EXPLORE_EPSILON,
                 lr=LEARNING_RATE, training=True):
        self.id = id
        self.V = np.ones(env.state_space_size)
        # state action value table, num_state*num_action_p1*num_action_p2
        self.Q = np.ones((env.state_space_size, env.action_space_size))
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
            # action = np.random.choice([i for i in range(self.action_num)], size=None, p=self.PI[state])
            action = np.argmax(self.Q[state])
        return action

    def update_Qval(self, a, o, s, next_s,reward):
        self.Q[s, a] = (1 - self.alpha) * self.Q[s, a] + self.alpha * (reward + self.gamma * self.V[next_s])

    def update_PI(self, state):
        opt_action = np.argmax(self.Q[state])
        self.V[state] = self.Q[state,opt_action]
        pass
    def save_agent(self,path = MODEL_PATH):
        # PI = pd.DataFrame(self.PI)
        Q = pd.DataFrame(self.Q)
        V = pd.DataFrame(self.V)
        # PI.to_csv(path + 'PI_0.txt', sep=' ', header=False, index=False)

        Q.to_csv(path + 'Q_'+str(self.id)+'.txt', sep=' ', header=False, index=False)
        V.to_csv(path + 'V_'+str(self.id)+'.txt', sep=' ', header=False, index=False)

    def load_agent(self,path = MODEL_PATH):
        # PI = pd.read_csv(path + 'PI_0.txt', sep=' ', header=None)
        Q = pd.read_csv(path + 'Q_'+str(self.id)+'.txt', sep=' ', header=None)
        V = pd.read_csv(path + 'V_'+str(self.id)+'.txt', sep=' ', header=None)
        # self.PI = PI.values
        self.Q = Q.values
        self.V = V.values
