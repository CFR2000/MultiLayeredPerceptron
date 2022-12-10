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
        
    def backward(self, inputs, labels):
        # Perform the backpropagation step
        # Comput the error at the output layer
        hidden = sigmoid(np.dot(inputs, self.weights1) + self.bias1)
        output = sigmoid(np.dot(hidden, self.weights2) + self.bias2)
        # Cost
        error = labels - output

        # Comput the error at the hidden layer
        hidden_error = np.dot(error, self.weights2.T)

        # Update the weights and biases of each layer
        self.weights2 += self.learning_rate * np.dot(hidden.T, error)
        self.bias2 += self.learning_rate * np.sum(error, axis=0)
        self.weights1 += self.learning_rate * np.dot(inputs.T, hidden_error)
        self.bias1 += self.learning_rate * np.sum(hidden_error, axis=0)

    def train(self, inputs, labels, epochs):
        # Train the network for a number of epochs
        for _ in range(epochs):
            self.backward(inputs, labels)

    def predict(self, inputs):
        # Use the trained network to make predictions on new data
        return self.forward(inputs)

        