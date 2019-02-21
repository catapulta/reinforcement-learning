import numpy as np


class MonteCarloControl:
    def __init__(self, game, eps, print_frequency):
        self.eps = eps
        self.print_frequency = print_frequency
        self.game = game
        self.q = np.zeros((20, 10, 2))
        self.n = np.zeros((20, 10, 2)).astype(int)
        self.r = 0
        self.a = []
        self.mean_reward = 0.
        self.n_win = 0
        self.n_sample = 0

    def sample(self):
        game = self.game()
        dealer = game.init_dealer_card
        player = game.player_sum
        state = dealer, player
        reward = None
        counter = np.zeros_like(self.n)
        terminal = 0
        while not terminal:
            # epsilon-greedy
            if np.random.random() < self.eps / (self.eps + self.n[player - 1, dealer - 1].sum()):
                action = np.random.choice((0, 1))  # exploration
            else:
                action = np.argmax(self.q[player - 1, dealer - 1])
            counter[player - 1, dealer - 1, action] = 1
            *state, reward, terminal = game.step(state, action)  # no need to sum rewards (only terminal)
            dealer, player = state

        self.n += counter
        self.q += np.divide((reward - self.q), self.n, out=np.zeros_like(self.q), where=counter != 0)

        # stats
        self.n_sample += 1
        self.mean_reward = self.mean_reward + 1 / (self.n.sum() + 1) * (reward - self.mean_reward)
        if reward == 1:
            self.n_win += 1

        if self.n_sample % self.print_frequency == 0:
            print("Sample %i, mean reward %.2f, wins %.2f" %
                  (self.n_sample, self.mean_reward, self.n_win / (self.n_sample + 1)))

        # if self.n_sample % 10000 == 0:
        #     from utils import plot
        #     plot(self.q, [0, 1])


if __name__ == '__main__':
    from easy21 import Easy21
    from utils import plotv
    import pickle

    mcc = MonteCarloControl(Easy21, 50, 10000)
    for i in range(500000):
        mcc.sample()

    pickle.dump(mcc.q, open('mcq.pkl', 'wb'))

    print('# stick')
    print(mcc.n[:, :, 0])
    print()
    print('# hit')
    print(mcc.n[:, :, 1])
    plotv(mcc.q, [0, 1])

