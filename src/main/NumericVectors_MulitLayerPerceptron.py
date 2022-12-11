import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import random
import pandas as pd
import matplotlib.pyplot as plt
import math

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
        # self.weights1 = np.random.normal(0.0, pow(hidden_size, -0.5), (hidden_size, input_size))
        # self.weights2 = np.random.normal(0.0, pow(output_size, -0.5), (output_size, hidden_size))
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights1 = np.random.normal(0.0, pow(hidden_size, -0.5), (hidden_size, input_size))
        self.weights2 = np.random.normal(0.0, pow(output_size, -0.5), (output_size, hidden_size))
        # Store the learning rate for the network
        # Store the learning rate for the network
        self.learning_rate = learning_rate
        self.activation = activation

    def product(self, x, y):
        return np.dot(x, y)

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
        z2 = sigmoid(output_in)
        return z1, z2

    def backward(self, inputs, labels):
        # Perform the backpropagation step
        # Compute the output of the network using the forward method
        hidden_val, output_val = self.forward(inputs)

        inputs = self.matrix_2d(inputs)
        labels = self.matrix_2d(labels)
        # Compute the error at the output layer
        error = labels - output_val
        # self.weights2 = self.weights2.T
        # Transpose the weights2 array so that it has the correct dimensions
        h_output_err = np.dot(self.weights2.T, error)
        ## Now tweak the network
        self.weights2 += self.learning_rate * np.dot((error * output_val * (1.0 - output_val)), hidden_val.T)
        self.weights1 += self.learning_rate * np.dot((h_output_err * hidden_val * (1.0 - hidden_val)), inputs.T)

    def train(self, inputs, labels):
        # Train the network for a number of epochs
        self.backward(inputs, labels)

    def predict(self, inputs):
        # Use the trained network to make predictions on new dat

        _, output_val = self.forward(inputs)
        return output_val

# ==================================================================== #
#                   End of class
# ==================================================================== #

#function to run training of model
def run_training(network, x, y, epochs):
    for epoch in range(epochs):
        for c, row in enumerate(x):
            inputs = (np.asfarray(row[:]))
            targets = (np.asfarray(y[c],dtype=float))
            network.train(inputs, targets)
        print(f"Epoch {epoch}: Training Occurring ...")
    pass


#function to run testing of model
def task2():

    def make_data():
        pass

    def run_training(network, x, y, epochs):
        for epoch in range(epochs):
            for c, row in enumerate(x):
                inputs = (np.asfarray(row[:]))
                targets = (np.asfarray(y[c],dtype=float))
                network.train(inputs, targets)
            print(f"Epoch {epoch}: Training Occurring ...")
        pass

    # #load the data set
    X = np.random.rand(500,4)
    y = np.zeros(500)

    for d in range(500): y[d] = math.sin((abs(X[d,0]-X[d,1]+X[d,2]-X[d,3])))

    #Separate in Train and Test Datasets
    X_train = X[0:400]
    y_train = y[0:400]

    X_test = X[400:501]
    y_test = y[400:501]

    #number of input, hidden and output nodes
    input_nodes = 4
    hidden_nodes = 5
    output_nodes = 1

    #learning rate
    learning_rate = 0.1
    activation = sigmoid


    #create the instance
    network = MultilayeredPerceptron(input_nodes, hidden_nodes, output_nodes, learning_rate, activation)

    epochs = 1000

    run_training(network, X_train, y_train, epochs)
    # for epoch in range(epochs):
    #     for counter, row in enumerate(X_train):
    #         inputs = (np.asfarray(row[:]))
    #         targets = (np.asfarray(y_train[counter],dtype=float))
    #         network.train(inputs, targets)
    #     print(f"Epoch {epoch}: Training Occurring ...")
    

    num_correct = []
    for count, row in enumerate(X_test):

        inputs = (np.asfarray(row[:]))
        y_pred = network.predict(inputs)

        true_l = y_test[count]
        pred_l = y_pred[0,0]
        
        if (round(true_l,1)== round(pred_l,1)): num_correct.append(1)
        else: num_correct.append(0)

    
    #calculate the performance score, the fraction of correct answers
    scorecard_array = np.asarray(num_correct)
    print("Testing Performance Accuracy = " , scorecard_array.sum() / scorecard_array.size)



if __name__ == "__main__":
    task2()

