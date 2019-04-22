from collections import defaultdict
import math


def dist_sort(t):
    return t[2]


def get_dist(left, right):
    return math.sqrt((right[0] - left[0])**2 + (right[1] - left[1])**2)


class GLOBALS:
    predator = 4
    prey = 2
    k = 5
    food = 2
    states = []
    knn = defaultdict(list)
    agents = []

    def calculate_knn(self):
        # 1. collect distances (minified by excluding 1 to 1 and bigger < smaller)
        # 1 to 1, 2 to 2 bad
        # 1 to 2 good, 2 to 1 bad
        # Store in format <node, node, distance>
        dist = []
        for i in range(len(self.agents)):
            for j in range(i, len(self.agents)):
                if i != j:
                    dist.append([i, j, get_dist(self.agents[i].state.p_pos, self.agents[j].state.p_pos)])

        # 2. sort on distances
        dist.sort(key=dist_sort)

        # 3. add connections to agents if the number of connections is < k
        for c in dist:
            if len(self.knn[c[0]]) < self.k:
                self.knn[c[0]].append(c[1])
            if len(self.knn[c[1]]) < self.k:
                self.knn[c[1]].append(c[0])


global_env = GLOBALS()
