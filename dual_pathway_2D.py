import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm
from functions import *

'''This model works, not the best, but it does'''

# 2D reward landscapes
def gaussian(coordinates, height, mean, spread):
    x, y = coordinates[0], coordinates[1]
    return height * np.exp(-((x-mean[0])**2 + (y-mean[1])**2)/(2*spread**2))
def reward_fn(coordinates):
    x, y = coordinates[0], coordinates[1]
    return np.exp(-((x-0.5)**2 + (y+0.2)**2))



# layer sizes
HVC_SIZE = 100
BG_SIZE = 50
RA_SIZE = 100 
MC_SIZE = 2
n_RA_clusters = 2
RA_cluster_size = RA_SIZE // n_RA_clusters

# sigmoid layer parameters
BG_sig_slope = 5  # lesser, slower the learning # BG sigmoidal should be as less steep as possible
BG_sig_mid = 0
RA_sig_slope = 30 # RA sigmoidal should be as steep as possible
RA_sig_mid = 0
MC_sig_slope = 10 # if lesser -> more difficult to climb the hill, assymptotes before 
MC_sig_mid = 0

# parameters
reward_window = 10
input = np.zeros(HVC_SIZE)
input[1] = 1

# Run paraneters
LEARING_RATE_RL = 0.1
LEARNING_RATE_HL = 5e-9
TRIALS = 1_00_000

# modes
HEBBIAN_LEARNING = True

# Model
class NN:
    def __init__(self, hvc_size, bg_size, ra_size, mc_size):
        self.W_hvc_bg = np.random.uniform(-1, 1, (hvc_size, bg_size)) # changing from -1 to 1 
        self.W_hvc_ra = np.zeros((hvc_size, ra_size)) # connections start from 0 and then increase
        self.W_bg_ra = np.random.uniform(0, 1, (bg_size, ra_size)) # const from 0 to 1
        self.W_ra_mc = np.random.uniform(0, 1, (ra_size, mc_size)) # const from 0 to 1
        # now we channelize RA to MC as x,y of MC should be independent of each other
        for i in range(n_RA_clusters):
            segPath = np.diag(np.ones(n_RA_clusters, int))[i] # diagonal 2x2 matrix
            self.W_ra_mc[i*RA_cluster_size : (i+1)*RA_cluster_size] *= segPath # set cross weights to zero
            
        self.hvc_size = hvc_size
        self.bg_size = bg_size
        self.ra_size = ra_size
        self.mc_size = mc_size  
        self.RA_cluster_size = RA_cluster_size
            
    def forward(self, hvc_array):
        self.hvc = hvc_array
        # count number of 1 in hvc, divide bg by that number
        num_ones = np.count_nonzero(hvc_array == 1)
        self.bg = new_sigmoid(np.dot(hvc_array/num_ones, self.W_hvc_bg) + np.random.normal(0, 0.05, self.bg_size), m = BG_sig_slope, a = BG_sig_mid)
        self.ra = new_sigmoid(np.dot(self.bg/self.bg_size, self.W_bg_ra) + np.dot(hvc_array/num_ones, self.W_hvc_ra)*HEBBIAN_LEARNING, m = RA_sig_slope, a = RA_sig_mid) 
        self.mc = new_sigmoid(np.dot(self.ra/self.RA_cluster_size, self.W_ra_mc), m = MC_sig_slope, a = MC_sig_mid)
        return self.mc

class Environment:
    def __init__(self, hvc_size, bg_size, ra_size, mc_size):
        self.hvc_size = hvc_size
        self.bg_size = bg_size
        self.ra_size = ra_size
        self.mc_size = mc_size
        self.model = NN(hvc_size, bg_size, ra_size, mc_size)
        self.rewards = []
        self.actions = []
        
    def get_reward(self, action):
        return reward_fn(action)
    
    def run(self, iterations, learning_rate, learning_rate_hl, input_hvc):
        for iter in tqdm(range(iterations)):
            # reward and baseline
            action = self.model.forward(input_hvc)
            reward = self.get_reward(action)
            self.rewards.append(reward)
            self.actions.append(action)
            reward_baseline = np.mean(self.rewards[-reward_window:])
            # Updates 
            # RL update
            dw_hvc_bg = learning_rate*(reward - reward_baseline)*input_hvc.reshape(self.hvc_size,1)*self.model.bg # RL update
            self.model.W_hvc_bg += dw_hvc_bg
            # HL update
            dw_hvc_ra = learning_rate_hl*input_hvc.reshape(self.hvc_size,1)*self.model.ra*HEBBIAN_LEARNING # lr is supposed to be much smaller here
            self.model.W_hvc_ra += dw_hvc_ra
            if iter % (TRIALS/100) == 0:    
                tqdm.write(f'Iteration: {iter}, Action: {action}, Reward: {reward}, Reward Baseline: {reward_baseline}')     
                
    # def plot_results(self):
    #     # plot rewards 
    #     plt.figure()
    #     plt.plot(self.rewards)
    #     plt.ylim(0, 1)
    #     plt.xlabel('Iterations')
    #     plt.ylabel('Reward')
    #     plt.title('Reward vs Iterations')
    #     plt.show()
        
    # def plot_trajectory(self):
    #     x, y = np.linspace(-2, 2, 50), np.linspace(-2, 2, 50)
    #     X, Y = np.meshgrid(x, y)
    #     Z = reward_fn([X, Y])
    #     # Create the contour plot
    #     plt.style.use('dark_background')
    #     plt.figure(figsize=(6, 6))
    #     plt.contour(X, Y, Z, levels=10)
    #     plt.title('Contour plot of reward function')
    #     plt.xlabel('x')
    #     plt.ylabel('y')
    #     plt.colorbar(label='Reward')
    #     # Extract x and y coordinates from trajectory
    #     x_traj, y_traj = zip(*self.actions)

    #     # Plot the trajectory as a line and starting point as a red circle
    #     plt.plot(x_traj[::10], y_traj[::10], '-b', label='Agent Trajectory')
    #     plt.scatter(x_traj[0], y_traj[0], c='red', label='Starting Point')  # Plot first point as red circle
    #     plt.legend()
    #     plt.show()
    
    def plot_results_and_trajectory(self):
        plt.style.use('dark_background')
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))

        # Plot rewards
        axs[0].plot(self.rewards)
        axs[0].set_ylim(0, 1)
        axs[0].set_xlabel('Iterations')
        axs[0].set_ylabel('Reward')
        axs[0].set_title('Reward vs Iterations')

        # Plot trajectory
        x, y = np.linspace(-2, 2, 50), np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)
        Z = reward_fn([X, Y])
        contour = axs[1].contour(X, Y, Z, levels=10)
        fig.colorbar(contour, ax=axs[1], label='Reward')
        x_traj, y_traj = zip(*self.actions)
        axs[1].plot(x_traj[::20], y_traj[::20], '-b', label='Agent Trajectory') # Plot every 20th point for efficiency
        axs[1].scatter(x_traj[0], y_traj[0], c='red', label='Starting Point')  # Plot first point as red circle
        axs[1].set_title('Contour plot of reward function')
        axs[1].set_xlabel('x')
        axs[1].set_ylabel('y')
        axs[1].legend()

        plt.tight_layout()
        plt.show()
        

env = Environment(HVC_SIZE, BG_SIZE, RA_SIZE, MC_SIZE)
env.run(TRIALS, LEARING_RATE_RL, LEARNING_RATE_HL, input)
env.plot_results_and_trajectory()