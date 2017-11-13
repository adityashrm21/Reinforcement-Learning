import numpy as np

# Assuming a grid environment env is available to us already
# Written using:
# 1. Algorithm mentioned in Sutton-Barto Reinforcement Learning: An Introduction
# 2. https://github.com/dennybritz/reinforcement-learning/blob/master/DP/Policy%20Evaluation%20Solution.ipynb

def iterativePolicyEvaluation(env, gamma = 1.0, epsilon = 0.0001, policy):
    V = np.zeros(env.noOfStates)
    while(true):
        delta = 0
        for s in range(env.noOfStates):
            v = 0
            for a, actionProbab in enumerate(policy[s]):
                for probab, reward, nextState in env.P[s][a]:

                    v += actionProbab * probab * (reward + gamma * V[nextState])
            delta = max(error, np.abs(v - V[s]))
            V[s] = v
        if(delta < epsilon):
            break
    return np.array(V)
    
