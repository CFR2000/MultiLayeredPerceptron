import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid(x):
    # Define the sigmoid activation function
    return 1 / (1+np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

class MultilayeredPerceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, activation):
        # Initialize the weights and biases for each layer of the network
        self.weights1 = np.random.normal(0.0, pow(hidden_size, -0.5), (hidden_size, input_size))
        self.weights2 = np.random.normal(0.0, pow(output_size, -0.5), (output_size, hidden_size))

        # self.weights1 = np.random.normal(0.0, pow(hidden_size, -0.5), (input_size, hidden_size))
        # self.weights2 = np.random.normal(0.0, pow(output_size, -0.5), (hidden_size, output_size))
        # Store the learning rate for the network
        self.learning_rate = learning_rate
        self.activation = activation
        pass

    def product(self, x, y):
        return np.dot(x,y)

    def matrix_2d(self, x):
        return np.array(x, ndmin=2).T

    def forward(self, inputs):
        # Perform the forward propagation step
        inputs = self.matrix_2d(inputs)

        #calculate values going into the hidden layer
        hidden_in = self.product(self.weights1, inputs)
        #passing hidden inputs thru activation to get weighted sum
        z1 = self.activation(hidden_in)
        #get weight sof values going into output layer
        output_in = self.product(self.weights2, z1)
        #getting value for the output layer
        z2 = self.activation(output_in)
        return z1, z2
        
    def backward(self, inputs, labels):
        # Perform the backpropagation step
        # Compute the output of the network using the forward method
        hidden_val, output_val = self.forward(inputs)

        inputs = self.matrix_2d(inputs)
        labels = self.matrix_2d(labels)
        # Compute the error at the output layer
        error = labels - output_val
        # Transpose the weights2 array so that it has the correct dimensions
        h_output_err = self.product(self.weights2.T, error)
        ## Now tweak the network
        self.weights2 += self.learning_rate * self.product(((error * output_val) * (1.0 - output_val)),hidden_val.T)
        self.weights1 += self.learning_rate * self.product(((h_output_err * hidden_val) * (1.0 - hidden_val)),inputs.T)

        pass

    def train(self, inputs, labels, epochs):
        # Train the network for a number of epochs
        output_val = []
        for epoch in range(epochs):

            # Print the current epoch number
            print(f"Epoch: {epoch}")

            # Perform the backpropagation step
            self.backward(inputs, labels)
            
            _, output_val = self.forward(inputs)
            # Print the output of the network after training
            print(f"Output: {output_val}")

    def predict(self, inputs):
        # Use the trained network to make predictions on new dat
        output_val=[]
        _, output_val = self.forward(inputs)
        return output_val



def train_and_test_and_save(network, X_train, y_train, X_test, y_test, epochs):
    # Train the network for the specified number of epochs
    for epoch in range(epochs):
        # Print the current epoch number
        print(f"Epoch: {epoch}")

        # Train the network on the training data
        network.backward(X_train, y_train)

        # Use the trained network to make predictions on the test data
        y_pred = network.predict(X_test)
        y_pred = np.round(y_pred)
        y_pred = y_pred.reshape(y_test.shape)

        # Compute the accuracy of the predictions
        accuracy = accuracy_score(y_test, y_pred)

        # Print the accuracy of the predictions
        print(f"Accuracy: {accuracy}")
    

    # Save the training and test scores to a text file
    with open("scores.txt", "w") as f:
        # Train the network on the training data
        for epoch in range(epochs):
            network.train(X_train, y_train, epoch)

            # Use the trained network to make predictions on the test data
            y_pred = network.predict(X_test)
            y_pred = np.round(y_pred)
            y_pred = y_pred.reshape(y_test.shape)

            # Compute the accuracy of the predictions
            accuracy = accuracy_score(y_test, y_pred)

            # Write the epoch and accuracy to the file
            f.write(f"Epoch: {epoch}, Accuracy: {accuracy}\n")


def task1():
    # Load the XOR dataset
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    #Initialize the network
    input_size = 2
    hidden_size = 4
    output_size = 1
    learning_rate = 0.1
    activation = sigmoid

    network = MultilayeredPerceptron(input_size, hidden_size, output_size, learning_rate, activation)

    # Test the network
    X_test = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_test = np.array([[0], [1], [1], [0]])


    epochs = 1000
    # train_and_test_and_save(network, X, y, X_test, y_test, epochs)

    network.train(X, y, epochs)
  

    y_pred = network.predict(X_test)

    # visualise_learning(output_data, epochs)

    y_pred = np.round(y_pred)
    y_pred = y_pred.reshape(y_test.shape)

    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)


if __name__ == "__main__":
    task1()


