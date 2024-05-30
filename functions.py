import numpy as np
import matplotlib.pyplot as plt

def new_sigmoid(x, m=0, a=0):
    """ Returns an output between -1 and 1 """
    return (2 / (1 + np.exp(-1*(x-a)*m))) - 1

def plot_reward_fn(reward_fn): # reward plotting function
    x = np.linspace(-1, 1, 100)
    y = [reward_fn(i) for i in x]
    plt.plot(x, y)
    plt.xlabel('Action')
    plt.ylabel('Reward')
    plt.title('Reward Landscape')
    plt.show()