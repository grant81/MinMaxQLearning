from minimax_q_agent import *
from soccer_env import *
import pickle
from random_agent import *
from q_agent import *
import matplotlib.pyplot as plt
env = SoccerEnv()
# for server runing pyplot
plt.switch_backend('agg')

def train(agent0, agent1, eps=10000):
    result = np.zeros(eps)
    # steps = np.zeros(eps)
    win_percs = []
    win_perc = 0
    for episode in range(eps):
        env.reset()
        reward = -1
        state0 = env.boardToState()
        state1 = env.boardToState(True)
        step = 0
        while reward < 0 and step < 800:
            step += 1
            a0 = agent0.take_action(state0)
            a1 = agent1.take_action(state1)
            next_state0, reward = env.step(a0, a1)
            next_state1 = env.boardToState(True)
            agent0.update_Qval(a0, a1, state0, next_state0, int(reward == 0))
            agent0.update_PI(state0)
            state0 = next_state0
            state1 = next_state1
        result[episode] = int(reward == 0)
        # steps[episode] = step
        if episode % 100 == 0 and episode > 0:
            c_win_perc = sum(result) / episode
            win_percs.append(c_win_perc)
            print('eposide: {}, win percentage: {}'.format(episode, c_win_perc))
            if c_win_perc > win_perc:
                win_perc = c_win_perc
                agent0.save_agent()
                file = open(MODEL_PATH + 'agent_obj.ag', 'wb')
                pickle.dump(agent0, file)
                file.close()

    plt.plot(win_percs)
    plt.legend(['win percentage'])
    plt.xlabel('hundreds episodes')
    plt.ylabel('win percentage')
    plt.savefig(MODEL_PATH+'training_plt.png')
   # plt.show()


def train_double(agent0, agent1, eps=10000):
    result0 = np.zeros(eps)
    result1 = np.zeros(eps)
    # steps = np.zeros(eps)
    win_percs = []
    win_perc = 0
    for episode in range(eps):
        env.reset()
        reward = -1
        state0 = env.boardToState()
        state1 = env.boardToState(True)
        step = 0
        while reward < 0 and step < 800:
            step += 1
            a0 = agent0.take_action(state0)
            a1 = agent1.take_action(state1)
            next_state0, reward = env.step(a0, a1)
            next_state1 = env.boardToState(True)
            agent0.update_Qval(a0, a1, state0, next_state0, int(reward == 0))
            agent0.update_PI(state0)
            agent1.update_Qval(a1, a0, state1, next_state1, int(reward == 1))
            agent1.update_PI(state1)
            state0 = next_state0
            state1 = next_state1
        result0[episode] = int(reward == 0)
        result1[episode] = int(reward == 1)

        if episode % 100 == 0 and episode > 0:
            c_win_perc = sum(result0) / episode
            c_win_perc1 = sum(result1)/episode
            win_percs.append(c_win_perc)
            print('eposide: {}, win percentage: {} vs {}'.format(episode, c_win_perc,c_win_perc1))
            if c_win_perc > win_perc:
                win_perc = c_win_perc
                agent0.save_agent()
                file = open(MODEL_PATH + 'agent_obj.ag', 'wb')
                pickle.dump(agent0, file)
                file.close()
    plt.plot(win_percs)
    plt.legend(['win percentage'])
    plt.xlabel('hundreds episodes')
    plt.ylabel('win percentage')
    plt.savefig(MODEL_PATH + 'training_plt.png')
    plt.show()

def test(agent0, agent1, num=100000):
    env = SoccerEnv()
    result = np.zeros(num)
    steps = np.zeros(num)

    for episode in range(num):
        env.reset()
        reward = -1
        state0 = env.boardToState()
        state1 = env.boardToState(True)
        step = 0
        while reward < 0 and step < 800:
            step += 1
            a0 = agent0.take_action(state0)
            a1 = agent1.take_action(state1)
            next_state0, reward = env.step(a0, a1)
            next_state1 = env.boardToState(True)
            state0 = next_state0
            state1 = next_state1
        result[episode] = int(reward == 0)
        steps[episode] = step
        if episode % 100 == 0 and episode > 0:
            c_win_perc = sum(result) / episode
            print('eposide: {}, win percentage: {}'.format(episode, c_win_perc))


# test()
# file = open(MODEL_PATH+'agent_obj.ag','rb')
# agent0 = pickle.load(file)
# file.close()
# agent0.training = False
agent0 = Q_Agent(0,env)
#agent0.load_agent('AGENTS/minimax_against_random/')
#agent0.training = False
agent1 = Q_Agent(1,env)
agent1.load_agent('AGENTS/q_against_random/')
agent1.training = False
#agent1 = Random_Agent(env)
# train_double(agent0,agent1,20000)
train(agent0,agent1,200000)
# test(agent1,agent0,100000)
