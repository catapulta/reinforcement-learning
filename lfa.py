import numpy as np


class LFA:
    def __init__(self, game, lmbda, eps, print_frequency):
        self.print_frequency = print_frequency
        self.game = game
        self.dealer_state = [[1, 4], [4, 7], [7, 10]]
        self.player_state = [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]
        self.action_set = [0, 1]
        self.feature_shape = (len(self.dealer_state), len(self.player_state), len(self.action_set))
        self.theta = np.random.randn(*self.feature_shape)  # q function params
        self.theta = np.ones(self.feature_shape)
        self.e = np.zeros(self.feature_shape)  # eligibility trace
        self.r = 0
        self.a = []
        self.mean_reward = 0.
        self.n_win = 0
        self.n_sample = 0
        self.lmbda = lmbda
        self.lr = 0.01  # learning rate
        self.eps = 0.05  # exploration rate
        self.learning = []
        self.map = None

    def epsilonGreedy(self, player, dealer):
        if np.random.random() < self.eps:
            action = np.random.choice((0, 1))  # exploration
        else:
            action = np.argmax([self.q(self.featurize(player, dealer, a)) for a in self.action_set])  # exploitation
        return action

    def featurize(self, player, dealer, action):
        assert 21 > player > 0, 'illegal player values: {}'.format(player)
        assert 10 >= dealer > 0, 'illegal dealer values: {}'.format(dealer)
        assert action == 1 or action == 0, 'illegal actions: {}'.format(action)
        x = np.zeros(self.feature_shape)

        ds = []
        for i, r in enumerate(self.dealer_state):
            if r[0] <= dealer <= r[1]:
                ds.append(i)

        ps = []
        for i, r in enumerate(self.player_state):
            if r[0] <= player <= r[1]:
                ps.append(i)

        assert len(ds) > 0 and len(ps) > 0, 'Failure to map from state to feature space.'

        for d in ds:
            for p in ps:
                x[d, p, action] = 1

        return x

    def mapQToStateSpace(self):
        map = np.zeros((20, 10, 2))

        for d in range(map.shape[1]):
            for p in range(map.shape[0]):
                for a in range(map.shape[2]):
                    x = self.featurize(p+1, d+1, a)
                    q = self.q(x)
                    map[p, d, a] = q
        self.map = map
        return map

    def q(self, x):
        return (x * self.theta).sum()

    def sample(self):
        game = self.game()
        dealer = game.init_dealer_card
        player = game.player_sum
        state = dealer, player
        self.e = np.zeros_like(self.e)
        terminal = 0
        while not terminal:
            action = self.epsilonGreedy(player, dealer)
            fstate = self.featurize(player, dealer, action)
            *state2, reward, terminal = game.step(state, action)
            dealer2, player2 = state2
            if not terminal:
                action2 = self.epsilonGreedy(player2, dealer2)
                fstate2 = self.featurize(player2, dealer2, action2)
                tderror = reward + self.q(fstate2) - self.q(fstate)
            else:
                tderror = reward - self.q(fstate)
            self.e += fstate  # add gradient of q wrt theta
            self.theta += self.e * tderror * self.lr
            self.e *= self.lmbda  # update past eligibility
            state = state2
            player = player2
            dealer = dealer2

        # stats
        self.n_sample += 1
        self.mean_reward = self.mean_reward + 1 / self.n_sample * (reward - self.mean_reward)
        if reward == 1:
            self.n_win += 1

        if self.n_sample % self.print_frequency == 0:
            print("Sample %i, mean reward %.2f, wins %.2f" %
                  (self.n_sample, self.mean_reward, self.n_win / (self.n_sample + 1)))
            self.learning.append([self.n_sample, self.mapQToStateSpace()])

        # if self.n_sample % 10000 == 0:
        #     from utils import plot
        #     plot(self.q, [0, 1])


if __name__ == '__main__':
    from easy21 import Easy21
    from utils import plotv, plotMSE, plotLearningCurve
    import pickle

    lfa = LFA(Easy21, .5, 100, 1000)
    for i in range(50000):
        lfa.sample()

    print(lfa.mapQToStateSpace()[:, :, 0])
    print()
    print(lfa.mapQToStateSpace()[:, :, 1])
    print()
    print(lfa.theta)
    print(lfa.q(lfa.featurize(20, 9, 1)))

    plotv(lfa.mapQToStateSpace(), [0, 1])

    mcq = pickle.load(open('mcq.pkl', 'rb'))

    mses = []
    learning = []
    lmbdas = np.linspace(0, 1, 11)
    for j, l in enumerate(lmbdas):
        print('Lambda:', l)
        lfa = LFA(Easy21, l, 100, 100)
        for i in range(10000):
            lfa.sample()

        mse = np.mean(np.square(mcq - lfa.mapQToStateSpace()))
        mses.append(mse)

        if j % 2 == 0:
            for n_sample, sarsaq in lfa.learning:
                mse = np.mean(np.square(mcq - sarsaq))
                learning.append([l, n_sample, mse])
    learning = np.array(learning)
    print(learning)
    plotMSE(mses, lmbdas)
    plotLearningCurve(learning)
