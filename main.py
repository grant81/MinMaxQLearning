from minimax_q_agent import *
from soccer_env import *
import pickle
from random_agent import *
from q_agent import *

env = SoccerEnv()


def train(agent0, agent1, eps=10000):
    result = np.zeros(eps)
    steps = np.zeros(eps)
    win_perc = 0
    for episode in range(eps):
        env.reset()
        reward = -1
        state = env.boardToState()
        step = 0
        while reward < 0 and step < 800:
            step += 1
            a0 = agent0.take_action(state)
            a1 = agent1.take_action(state)
            next_state, reward = env.step(a0, a1)
            agent0.update_Qval(a0, a1, state, next_state, int(reward == 0))
            agent0.update_PI(state)
            state = next_state
        result[episode] = int(reward == 0)
        steps[episode] = step
        if episode % 100 == 0 and episode > 0:
            c_win_perc = sum(result) / episode
            print('eposide: {}, win percentage: {}'.format(episode, c_win_perc))
            if c_win_perc > win_perc:
                win_perc = c_win_perc
                agent0.save_agent()
                file = open(MODEL_PATH + 'agent_obj.ag', 'wb')
                pickle.dump(agent0, file)
                file.close()


def train_double(agent0, agent1, eps=10000):
    result0 = np.zeros(eps)
    result1 = np.zeros(eps)
    # steps = np.zeros(eps)
    win_perc = 0
    for episode in range(eps):
        env.reset()
        reward = -1
        state = env.boardToState()
        step = 0
        while reward < 0 and step < 800:
            step += 1
            a0 = agent0.take_action(state)
            a1 = agent1.take_action(state)
            next_state, reward = env.step(a0, a1)
            agent0.update_Qval(a0, a1, state, next_state, int(reward == 0))
            agent0.update_PI(state)
            agent1.update_Qval(a1, a0, state, next_state, int(reward == 0))
            agent1.update_PI(state)
            state = next_state
        result0[episode] = int(reward == 0)
        result1[episode] = int(reward == 1)

        if episode % 100 == 0 and episode > 0:
            c_win_perc = sum(result0) / episode
            c_win_perc1 = sum(result1)/episode
            print('eposide: {}, win percentage: {} vs {}'.format(episode, c_win_perc,c_win_perc1))
            if c_win_perc > win_perc:
                win_perc = c_win_perc
                agent0.save_agent()
                file = open(MODEL_PATH + 'agent_obj.ag', 'wb')
                pickle.dump(agent0, file)
                file.close()


def test(agent0, agent1, num=100000):
    env = SoccerEnv()
    result = np.zeros(num)
    steps = np.zeros(num)
    for episode in range(num):
        env.reset()
        reward = -1
        state = env.boardToState()
        step = 0
        while reward < 0 and step < 800:
            step += 1
            a0 = agent0.take_action(state)
            a1 = agent1.take_action(state)
            next_state, reward = env.step(a0, a1)
            state = next_state
        result[episode] = int(reward == 0)
        steps[episode] = step
        if episode % 100 == 0 and episode > 0:
            c_win_perc = sum(result) / episode
            print('eposide: {}, win percentage: {}'.format(episode, c_win_perc))


# test()
file = open('AGENTS/minimax_against_random/'+'agent_obj.ag','rb')
agent0 = pickle.load(file)
file.close()
agent0.training = False
agent0.id = 1
agent1 = Random_Agent(env)
# agent0 = MiniMax_Q_Agent(0, env)
test(agent0,agent1,100000)
