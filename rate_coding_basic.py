import numpy as np
import matplotlib.pyplot as plt

# Here we plan to make a simple rate coding model
INPUT_SIZE = 10
OUTPUT_SIZE = 1

# Define the model
class RateCodingModel:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(output_size)
        self.learning_rate = 0.0001
        
    def forward(self, x):
        return np.dot(x, self.weights) + self.bias

    def backward(self, x, y):
        y_pred = self.forward(x)
        error = y - y_pred
        self.weights += self.learning_rate * x.reshape(self.input_size,1) * error.reshape(1,self.output_size)
        self.bias += self.learning_rate * error
        
    def predict(self, x):
        return np.tanh(self.forward(x))
    
# Define the environment
class Environment:
    def __init__(self, input_size, output_size, target):
        self.input_size = input_size
        self.output_size = output_size
        self.target = target
        self.loss_array = []
        self.output_array = []
        self.model = RateCodingModel(input_size, output_size)
        
    def get_input(self):
        return np.random.rand(self.input_size)
    
    def run(self, episodes):
        for episode in range(episodes):
            x = self.get_input()
            y = self.target
            self.model.backward(x, y)
            if episode % 100 == 0:
                print(f'Episode: {episode}, Loss: {np.abs(y - self.model.predict(x))}')
                self.loss_array.append(np.abs(y - self.model.predict(x)))
                self.output_array.append(self.model.predict(x))
    
    def plot_loss(self):
        plt.figure()
        plt.plot(self.loss_array)
        plt.xlabel('Episodes')
        plt.ylabel('Loss')
        plt.title('Loss vs Episodes')
        plt.show()
    
    def plot_output(self):
        plt.figure()
        plt.plot(self.output_array)
        plt.plot(np.ones(len(self.output_array)) * self.target, 'r--')
        plt.xlabel('Episodes')
        plt.ylabel('Output')
        plt.title('Output vs Episodes')
        plt.show()
                
# Run the environment
env = Environment(INPUT_SIZE, OUTPUT_SIZE, 0.4)
env.run(10000)
env.plot_loss()
env.plot_output()
# print(env.model.weights)