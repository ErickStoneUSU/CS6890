from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

import BlackJack


class Monte:
    bj = BlackJack.BlackjackEnv()
    count = 0
    episode_count = 0
    # defaultdict does insert if accessor is not found
    # much better than throwing a key exception
    q = defaultdict(lambda: np.zeros(2))
    pi = defaultdict(lambda: np.zeros(2))
    iterations = 100000
    explore = 8000
    delta = .1
    learning_rate = .1

    def monte_carlo(self):
        for i in range(1, self.iterations):
            # list of [S,A,R]
            ep = self.build_episode()
            self.update_q(ep)

        pi = dict((k, np.argmax(v)) for k, v in self.q.items())

        # figure with subplots
        fig = plt.figure()
        self.plot(pi, fig, True, 223)
        self.plot(pi, fig, False, 224)
        plt.show()

    # work through a full game
    # collect the history of actions,rewards
    # Monte Carlo approximates the best way to win based on lots of experience
    def build_episode(self):
        ep = []
        # random state in bj environment is observed
        s = self.bj.reset()
        self.episode_count = self.episode_count + 1

        while True:
            # choose random action either new sample
            # or potentially new iteration of old sample
            # q shows the highest quality (probabilistic) policy
            a = None
            if s in self.q:
                t_a = np.argmax(self.q[s])
                t = np.zeros(2) + self.delta
                t[t_a] = 1 - self.delta
                self.delta = self.delta * self.learning_rate
                a = np.random.choice(a=np.arange(2), p=t)
            else:
                a = self.bj.action_space.sample()

            # step takes the action on the environment
            # producing the next state, reward, and game completed status
            s_next, r, finished, _ = self.bj.step(a)
            ep.append((s, a, r))
            s = s_next

            if finished:
                break
        return ep

    def update_q(self, ep):
        s, a, r = zip(*ep)
        count = 1
        for i, s_t in enumerate(reversed(s)):
            if not r:
                self.q[s_t][a[i]] = self.q[s_t][a[i]] + (sum(r[i:]) - self.q[s_t][a[i]]) / (count + 1) - 1
            else:
                self.q[s_t][a[i]] = self.q[s_t][a[i]] + (sum(r[i:]) - self.q[s_t][a[i]]) / (count + 1) + 1

    def plot(self, policy, fig, usable_ace, subplot):
        sub = fig.add_subplot(subplot)
        if usable_ace:
            sub.set_title('Usable Ace')
        else:
            sub.set_title('No usable Ace')

        pi = np.ones(121).reshape(11, 11)

        for d_card in range(0, 11):
            for p_hand in range(0, 11):
                if (p_hand + 11, d_card, usable_ace) in policy:
                    try:
                        pi[d_card, p_hand] = policy[p_hand + 11, d_card, usable_ace]
                    except:
                        print(p_hand)
                        print(d_card)
                else:
                    try:
                        pi[d_card, p_hand] = 1
                    except:
                        print(p_hand)
                        print(d_card)

        cmap = colors.ListedColormap(['red', 'blue'])
        grid = sub.imshow(np.rot90(pi)[:, 1:], cmap=cmap, vmin=0, vmax=1, extent=[0.5, 10.5, 10.5, 21.5])

        plt.xticks(range(1, 11), ('A', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
        plt.yticks(range(11, 22))

        # label
        sub.set_xlabel('Dealer Shown Card')
        sub.set_ylabel('Player Hand Sum')

        # use grid lines
        sub.grid(linestyle='-', linewidth=2)

        # the color bar indicator
        divider = make_axes_locatable(sub)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = plt.colorbar(grid, ticks=[0, 1], cax=cax)
        cbar.ax.set_yticklabels(['HIT', 'STAY'])
        cbar.ax.invert_yaxis()


var = Monte()
var.monte_carlo()
