import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def reward_fn(coordinates):
    x, y = coordinates[0], coordinates[1]
    return np.exp(-((x)**2 + (y)**2))

def plot_contour(reward_fn):
    x, y = np.linspace(-2, 2, 50), np.linspace(-2, 2, 50)
    X, Y = np.meshgrid(x, y)
    Z = reward_fn([X, Y])
    # Create the contour plot
    plt.style.use('dark_background')
    plt.figure(figsize=(6, 6))
    plt.contour(X, Y, Z, levels=50)
    plt.title('Contour plot of reward function')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='Reward')
    
class ClimberModel:
    def __init__(self, start_position=(-1, 1)):
        self.position = start_position

    def move(self, step_size, max_x=2, max_y=2):
        # Generate random direction (consider using more sophisticated strategies)
        theta = np.random.uniform(0, 2*np.pi)
        dx = step_size * np.cos(theta)
        dy = step_size * np.sin(theta)

        # Limit movement within bounds
        new_x = min(max(self.position[0] + dx, -max_x), max_x)
        new_y = min(max(self.position[1] + dy, -max_y), max_y)

        self.position = (new_x, new_y)

# Simulation parameters
num_steps = 1000
step_size = 0.1

# Create climber model and trajectory list
climber = ClimberModel()  # Customize starting position
trajectory = []

# Run simulation
for _ in tqdm(range(num_steps)):
    climber.move(step_size)
    trajectory.append(climber.position)

# Plot the contour
plot_contour(reward_fn)

# Extract x and y coordinates from trajectory
x_traj, y_traj = zip(*trajectory)

# Plot the trajectory as a line and starting point as a red circle
plt.plot(x_traj, y_traj, '-b', label='Agent Trajectory')
plt.scatter(x_traj[0], y_traj[0], c='red', label='Starting Point')  # Plot first point as red circle
plt.legend()
plt.show()