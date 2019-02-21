import numpy as np


class SARSA:
    def __init__(self, game, lmbda, eps, print_frequency):
        self.eps = eps
        self.print_frequency = print_frequency
        self.game = game
        self.q = np.zeros((20, 10, 2))  # q function
        self.n = np.zeros((20, 10, 2)).astype(int)  # state-action counter
        self.e = np.zeros((20, 10, 2))  # eligibility trace
        self.r = 0
        self.a = []
        self.mean_reward = 0.
        self.n_win = 0
        self.n_sample = 0
        self.lmbda = lmbda
        self.learning = []

    def epsilonGreedy(self, player, dealer):
        if np.random.random() < self.eps / (self.eps + self.n[player - 1, dealer - 1].sum()):
            action = np.random.choice((0, 1))  # exploration
        else:
            action = np.argmax(self.q[player - 1, dealer - 1])  # exploitation
        return action

    def sample(self):
        game = self.game()
        dealer = game.init_dealer_card
        player = game.player_sum
        state = dealer, player
        self.e = np.zeros_like(self.e)
        terminal = 0
        while not terminal:
            action = self.epsilonGreedy(player, dealer)
            *state2, reward, terminal = game.step(state, action)
            dealer2, player2 = state2
            if not terminal:
                action2 = self.epsilonGreedy(player2, dealer2)
                tderror = reward + self.q[player2 - 1, dealer2 - 1, action2] - self.q[player - 1, dealer - 1, action]
            else:
                tderror = reward - self.q[player - 1, dealer - 1, action]
            self.n[player - 1, dealer - 1, action] += 1
            self.e[player - 1, dealer - 1, action] += 1  # make current state eligible
            self.q += np.divide(self.e * tderror, self.n, out=np.zeros_like(self.q), where=self.e != 0)
            self.e *= self.lmbda  # update past eligibility
            state = state2
            player = player2
            dealer = dealer2

        # stats
        self.n_sample += 1
        self.mean_reward = self.mean_reward + 1 / (self.n.sum() + 1) * (reward - self.mean_reward)
        if reward == 1:
            self.n_win += 1

        if self.n_sample % self.print_frequency == 0:
            print("Sample %i, mean reward %.2f, wins %.2f" %
                  (self.n_sample, self.mean_reward, self.n_win / (self.n_sample + 1)))
            self.learning.append([self.n_sample, self.q.copy()])

        # if self.n_sample % 10000 == 0:
        #     from utils import plot
        #     plot(self.q, [0, 1])


if __name__ == '__main__':
    from easy21 import Easy21
    from utils import plotv, plotMSE, plotLearningCurve
    import pickle

    sarsa = SARSA(Easy21, .5, 100, 1000)
    for i in range(50000):
        sarsa.sample()

    print('# stick')
    print(sarsa.n[:, :, 0])
    print()
    print('# hit')
    print(sarsa.n[:, :, 1])
    plotv(sarsa.q, [0, 1])

    mcq = pickle.load(open('mcq.pkl', 'rb'))

    mses = []
    learning = []
    lmbdas = np.linspace(0, 1, 11)
    for j, l in enumerate(lmbdas):
        sarsa = SARSA(Easy21, l, 100, 100)
        for i in range(1000):
            sarsa.sample()

        mse = np.mean(np.square(mcq - sarsa.q))
        mses.append(mse)

        if j % 2 == 0:
            for n_sample, sarsaq in sarsa.learning:
                mse = np.mean(np.square(mcq - sarsaq))
                learning.append([l, n_sample, mse])
    learning = np.array(learning)
    plotMSE(mses, lmbdas)
    plotLearningCurve(learning)
