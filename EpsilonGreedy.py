'''
Consider the following learning problem. You are faced repeatedly with a choice
among k different options, or actions. After each choice you receive a numerical
reward chosen from a stationary probability distribution that depends on the action
you selected. Your objective is to maximize the expected total reward over some
time period, for example, over 1000 action selections, or time steps.
This is the original form of the k-armed bandit problem.
'''

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
