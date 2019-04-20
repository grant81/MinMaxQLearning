import numpy as np


class SoccerEnv:
    def __init__(self, h=4, w=5, p0_pos=[1, 2], p1_pos=[3, 4], ball_holder=-1):
        self.h = h
        self.w = w
        self.positions = np.array([p0_pos, p1_pos])
        # the goals are at center of each side
        self.goal_pos1 = np.array([int(h / 2), 0])
        self.goal_pos2 = np.array([int(h / 2), w - 1])
        self.goal_positions = np.array([self.goal_pos1, self.goal_pos2])
        self.state_space_size = h * w * (h * w - 1) * 2
        self.action_space_size = 5

        if ball_holder < 0 or ball_holder > 1:
            self.ball_holder = self.select_player_random()
        else:
            self.ball_holder = ball_holder

    def reset(self, p0_pos=[0, 0], p1_pos=[0, 4], ball_holder=-1):
        self.positions = np.array([p0_pos, p1_pos])
        if ball_holder < 0 or ball_holder > 1:
            self.ball_holder = self.select_player_random()
        else:
            self.ball_holder = ball_holder

    def step(self, action0, action1):
        first_player_to_move = self.select_player_random()
        goal = -1
        if first_player_to_move == 0:
            goal = self.move(0, action0)
            if goal < 0:
                self.move(1, action1)
        else:
            goal = self.move(1, action1)
            if goal < 0:
                self.move(0, action0)
        # return state and reward
        encoded_state = self.boardToState()
        return encoded_state, goal

    def move(self, player, action):
        opponent = 1 - player
        newPosition = self.positions[player] + self.action_decoder(action)
        # make sure it is in the field
        if newPosition[0] >= self.h:
            newPosition[0] = self.h - 1
        elif newPosition[1] >= self.w:
            newPosition[1] = self.w - 1
        if newPosition[0]<0:
            newPosition[0] =0
        if newPosition[1]<0:
            newPosition[1] = 0
        if np.array_equiv(newPosition, self.positions[opponent]):
            self.ball_holder = opponent
            return -1
        # check if the player scored
        elif self.ball_holder == player and np.array_equiv(newPosition, self.goal_positions[opponent]):
            self.positions[player] = newPosition
            return 1 * player
        else:
            self.positions[player] = newPosition
            return -1

    def boardToState(self,reverse_order=False):
        if reverse_order:
            xA, yA = self.positions[1]
            xB, yB = self.positions[0]
            ball_holder = int(not self.ball_holder)
        else:
            xA, yA = self.positions[0]
            xB, yB = self.positions[1]
            ball_holder = self.ball_holder
        sA = xA * self.w + yA
        sB = xB * self.w + yB
        sB -= 1 if sB > sA else 0
        # encoded state on the bases on w*h-1, holding' Sa' Sb
        state = (sA * (self.w * self.h - 1) + sB) + (self.w * self.h) * (self.w * self.h - 1) * ball_holder
        return state

    def action_decoder(self, action):
        switcher = {
            0: [-1, 0],
            1: [0, 1],
            2: [1, 0],
            3: [0, -1],
            4: [0, 0],
        }
        return switcher[action]

    def select_player_random(self):
        return np.random.randint(0, 2)

    def render(self):
        game = ''
        row_count = 0
        col_count = 0
        for y in range(-1, self.h):
            for x in range(-1, self.w):
                if x == -1 and y == -1:
                    game += '+'
                else:
                    if x == -1:
                        game += str(row_count)
                        row_count += 1
                    elif y == -1:
                        game += str(col_count)
                        col_count += 1
                    else:
                        if np.array_equiv(np.array([y, x]), self.positions[0]):
                            game += 'A' if self.ball_holder is 0 else 'a'
                        elif np.array_equiv(np.array([y, x]), self.positions[1]):
                            game += 'B' if self.ball_holder is 1 else 'b'
                        elif np.array_equiv(np.array([y, x]), self.goal_positions[0]):
                            game += '|'
                        elif np.array_equiv(np.array([y, x]), self.goal_positions[1]):
                            game += '|'
                        else:
                            game += '-'
            game += '\n'
        print(game)


# s = SoccerEnv()
# s.render()
