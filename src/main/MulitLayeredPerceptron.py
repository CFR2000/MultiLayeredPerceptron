import numpy as np

def sigmoid(x):
    # Define the sigmoid activation function
    return 1 / (1+np.exp(-x))

class MultilayeredPerceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        # Initialize the weights and biases for each layer of the network
        self.weights1 = np.random.rand(input_size, hidden_size) * 0.1
        self.bias1 = np.random.rand(hidden_size) * 0.1
        self.weights2 = np.random.rand(hidden_size, output_size) * 0.1
        self.bias2 = np.random.rand(output_size) * 0.1

        # Store the learning rate for the network
        self.learning_rate = learning_rate
        pass

    def forward(self, inputs):
        # Perform the forward propagataion step
        hidden = sigmoid(np.dot(inputs, self.weights1) + self.bias1)
        output = sigmoid(np.dot(hidden, self.weights2) + self.bias2)
        return output
        
