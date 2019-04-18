from minimax_q_agent import *
from soccer_env import *
import pickle
env = SoccerEnv()
minmax_agent0 = MiniMax_Q_Agent(0,env)
minmax_agent1 = MiniMax_Q_Agent(0,env)
result = np.zeros(1000)
steps = np.zeros(1000)
win_perc = 0
for episode in range(10000):
    env.reset()
    reward = -1
    state = env.boardToState()
    step = 0
    while reward <0 and step<1000:
        step += 1
        a0 = minmax_agent0.take_action(state)
        a1 = np.random.randint(0,5)
        state,reward = env.step(a0,a1)
        minmax_agent0.update_Qval(a0,a1,state,int(reward == 0))
        minmax_agent0.update_PI(state)
        # minmax_agent1.update_Qval(a1, a0, state, int(reward == 1))
        # minmax_agent1.update_PI(state)
    result[episode] = int(reward==0)
    steps[episode] = step
    if episode %100 == 0 and episode >0:
        c_win_perc = sum(result) / episode
        print('eposide: {}, win percentage: {}'.format(episode,c_win_perc))
        if c_win_perc > win_perc:
            win_perc = c_win_perc
            minmax_agent0.save_agent()
            pickle.dump(minmax_agent0,MODEL_PATH+'agent_obj.ag')