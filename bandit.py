import numpy as np

class EpsilonGreedy():
    
    def __init__(self, epsilon, N, Q):
        self.epsilon = epsilon
        self.N = N
        self.Q = Q
        return

    def initialize(self, k):
        self.N = [0 for col in range(k)]
        self.Q = [0.0 for col in range(k)]
        return
        
    def exploitOrExplore(self, k):
        rand = random.random()
        if rand > self.epsilon:
            return np.argmax(self.Q)
        else:
            return np.random.randint(k)

            
    def update(self, chosen_arm, reward):
        self.N[chosen_arm] = self.N[chosen_arm] + 1
        n = self.N[chosen_arm]

        value = self.Q[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.Q[chosen_arm] = new_value
        return