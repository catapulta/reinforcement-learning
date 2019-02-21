import random


class Easy21:
    def __init__(self):
        self.black = 2/3
        self.dealer = int(random.uniform(1, 11))
        self.init_dealer_card = self.dealer
        self.player_sum = int(random.uniform(1, 11))

    def step(self, state, action):
        self.dealer, self.player_sum = state
        if action == 1:  # hit
            self.player_sum += self.draw()
            if self.player_sum >= 21 or self.player_sum < 1:
                return self.init_dealer_card, self.player_sum, -1, 1
            else:
                return self.init_dealer_card, self.player_sum, 0, 0
        elif action == 0:  # stick
            return self.dealer_draw()

    def draw(self):
        black = 1 if random.uniform(0, 1) >= self.black else -1
        card = int(random.uniform(1, 10))
        return black * card

    def dealer_draw(self):
        while self.dealer < 17:
            self.dealer += self.draw()
            if self.dealer >= 21 or self.dealer < 1:
                return self.init_dealer_card, self.player_sum, 1, 1

        reward = 1 if self.player_sum > self.dealer else (0 if self.player_sum == self.dealer else -1)

        return self.init_dealer_card, self.player_sum, reward, 1


if __name__ == '__main__':
    e21 = Easy21()
    print(e21.step([10, 10], 1))
    # e = 0
    # n = 0
    # while e < 2:
    #     e21 = Easy21()
    #     n += 1
    #     e = e21.dealer_draw()[2]
    # print(n)

