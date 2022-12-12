# ==================================================================== #
#                         Libraries                                    #
# ==================================================================== #


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import random
import pandas as pd
import matplotlib.pyplot as plt
import math
from visualise import confusion_visual, plot_errors
from sklearn.model_selection import GridSearchCV


# ==================================================================== #
#                    Activation Funcitons                              #
# ==================================================================== #


def sigmoid(x):
    # Define the sigmoid activation function
    return 1 / (1+np.exp(-x))


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


# ==================================================================== #
#                   Start Of Neural Network                            #
# ==================================================================== #


class MultilayeredPerceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate, activation):
        # Initialize the weights and biases for each layer of the network
        # self.weights1 = np.random.normal(0.0, pow(hidden_size, -0.5), (hidden_size, input_size))
        # self.weights2 = np.random.normal(0.0, pow(output_size, -0.5), (output_size, hidden_size))
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.weights1 = np.random.normal(
            0.0, pow(hidden_size, -0.5), (hidden_size, input_size))
        self.weights2 = np.random.normal(
            0.0, pow(output_size, -0.5), (output_size, hidden_size))
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

        # calculate values going into the hidden layer
        hidden_in = self.product(self.weights1, inputs)
        # passing hidden inputs thru activation to get weighted sum
        z1 = self.activation(hidden_in)
        # get weight sof values going into output layer
        output_in = self.product(self.weights2, z1)
        # getting value for the output layer
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
        # Now tweak the network
        self.weights2 += self.learning_rate * \
            np.dot((error * output_val * (1.0 - output_val)), hidden_val.T)
        self.weights1 += self.learning_rate * \
            np.dot((h_output_err * hidden_val * (1.0 - hidden_val)), inputs.T)

    def train(self, inputs, labels):
        # Train the network for a number of epochs
        self.backward(inputs, labels)

    def predict(self, inputs):
        # Use the trained network to make predictions on new dat

        _, output_val = self.forward(inputs)
        return output_val


# ==================================================================== #
#               Methods for running Training and Testing               #
# ==================================================================== #


# function to run training of model
def run_training(network, x, y, epochs, display_output=True):
    for epoch in range(epochs):
        for c, row in enumerate(x):
            inputs = (np.asfarray(row[:]))
            targets = (np.asfarray(y[c], dtype=float))
            network.train(inputs, targets)
        if display_output:
            print(f"Epoch {epoch}: Training Occurring ...")
    pass



# function to run testing of model

def run_testing(network, x, y, type='Test'):
    num_correct = []
    true_labels = []
    predicted_labels = []
    for count, row in enumerate(x):

        inputs = (np.asfarray(row[:]))
        y_pred = network.predict(inputs)

        true_l = y[count]
        pred_l = y_pred[0, 0]
        true_labels.append(round(true_l, 1))
        predicted_labels.append(round(pred_l, 1))

        if (round(true_l, 1) == round(pred_l, 1)):
            num_correct.append(1)
        else:
            num_correct.append(0)

    with open(f"scores_task2_{count+1}_{type}.txt", "w") as f:
        correct_array = np.asarray(num_correct)
        performance = correct_array.sum() / correct_array.size
        print(f"\nTesting {type} Performance = {performance}")
        f.write(f"\nTesting {type} Performance = {performance}")
        evaluate(true_labels, predicted_labels)
    pass

# ==================================================================== #
#           Helper Functions, grabbing values for Evaluation           #
# ==================================================================== #


def get_pred_values(network, x, y):
    true_labels = []
    predicted_labels = []
    for count, row in enumerate(x):
        inputs = (np.asfarray(row[:]))
        y_pred = network.predict(inputs)

        true_l = y[count]
        pred_l = y_pred[0, 0]
        true_labels.append(round(true_l, 1))
        predicted_labels.append(round(pred_l, 1))
    return np.array(true_labels), np.array(predicted_labels)

def get_pred_values2(network, x, y):
    true_labels = []
    predicted_labels = []
    for count, row in enumerate(x):
        inputs = (np.asfarray(row[:]))
        y_pred = network.predict(inputs)

        true_l = y[count]
        pred_l = y_pred[0, 0]
        true_labels.append(true_l)
        predicted_labels.append(pred_l)
    return np.array(true_labels), np.array(predicted_labels)


# Digitising the predictions and truths and gets the scores for the model.
def evaluate(true, pred):
    # Digitize T,P values
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    true_discrete = np.digitize(true, bins)
    pred_discrete = np.digitize(pred, bins)

    #Evaluate Performance
    print("Accuracy Score = ", accuracy_score(true_discrete, pred_discrete))
    print("Precision Score = ", precision_score(true_discrete, pred_discrete,
          average='micro'))  # micro suitable for multiclass analysis
    print("Recall Score = ", recall_score(true_discrete, pred_discrete,
          average='micro'))  # micro suitable for multiclass classification
    # micro suitable for multiclass classification
    print("F1 Score = ", f1_score(true_discrete, pred_discrete, average='micro'))
    pass

# ==================================================================== #
#                   Generate Data for Model                            #
# ==================================================================== #


def generate_data(rows, cols, train_size):
    # #load the data set
    X = np.random.rand(rows, cols)
    y = np.zeros(rows)

    for d in range(rows):
        y[d] = math.sin((abs(X[d, 0]-X[d, 1]+X[d, 2]-X[d, 3])))

    train_size = int(rows*train_size)
    # Separate in Train and Test Datasets
    X_train = X[0:train_size]  # 500*.8
    y_train = y[0:train_size]  # 500*.8

    X_test = X[train_size:(rows+1)]  # 500 + 1
    y_test = y[train_size:(rows+1)]  # 500 + 1
    return X_train, y_train, X_test, y_test


pass



# ==================================================================== #
#                       TESTING SUITIES                                #
# ==================================================================== #
def task2_100():

    # load the data set.
    X_train, y_train, X_test, y_test = generate_data(
        rows=500, cols=4, train_size=0.8)
    # Number of epochs the network will be trained on.
    epochs = 20
    # Create a network.
    network = MultilayeredPerceptron(
        input_size=4, hidden_size=40, output_size=1, learning_rate=0.1, activation=sigmoid)
    # Train network.
    run_training(network, X_train, y_train, epochs)
    # Test Network train Performance.
    run_testing(network, X_train, y_train, type='Train')
    # Test Network testing Performance.
    run_testing(network, X_test, y_test)
    # get true and predicitons
    true_labels, predicted_labels = get_pred_values(network, X_test, y_test)
    # visualise predictions
    # confusion_visual(true_labels, predicted_labels, epochs)
    # Visualise Errors 
    # plot_errors(true_labels, predicted_labels, epochs)


def task2_1000():

    # load the data set.
    X_train, y_train, X_test, y_test = generate_data(
        rows=500, cols=4, train_size=0.8)
    # Number of epochs the network will be trained on.
    epochs = 100
    # Create a network.
    network = MultilayeredPerceptron(
        input_size=4, hidden_size=40, output_size=1, learning_rate=0.1, activation=sigmoid)
    # Train network.
    run_training(network, X_train, y_train, epochs)
    
    # Test Network Performance.
    run_testing(network, X_train, y_train, type='Train')
    run_testing(network, X_test, y_test)
    # get true and predicitons
    true_labels, predicted_labels = get_pred_values(network, X_test, y_test)
    # visualise predictions
    confusion_visual(true_labels, predicted_labels, epochs)
    # Visualise Errors 
    # plot_errors(true_labels, predicted_labels, epochs)


def task2_10000():

    # load the data set.
    X_train, y_train, X_test, y_test = generate_data(
        rows=500, cols=4, train_size=0.8)
    # Number of epochs the network will be trained on.
    epochs = 10000
    # Create a network.
    network = MultilayeredPerceptron(
        input_size=4, hidden_size=40, output_size=1, learning_rate=0.1, activation=sigmoid)
    # Train network.
    run_training(network, X_train, y_train, epochs)
    # Test Network Performance.
    run_testing(network, X_train, y_train, type='Train')
    run_testing(network, X_test, y_test)
    # get true and predicitons
    true_labels, predicted_labels = get_pred_values2(network, X_test, y_test)
    # visualise predictions
    confusion_visual(true_labels, predicted_labels, epochs)
    # Visualise Errors 
    # plot_errors(true_labels, predicted_labels, epochs)

def task2_25000():

    # load the data set.
    X_train, y_train, X_test, y_test = generate_data(
        rows=500, cols=4, train_size=0.8)
    # Number of epochs the network will be trained on.
    epochs = 25000
    # Create a network.
    network = MultilayeredPerceptron(
        input_size=4, hidden_size=40, output_size=1, learning_rate=0.1, activation=sigmoid)
    # Train network.
    run_training(network, X_train, y_train, epochs)
    # Test Network Performance.
    run_testing(network, X_train, y_train, type='Train')
    run_testing(network, X_test, y_test)
    # get true and predicitons
    true_labels, predicted_labels = get_pred_values(network, X_test, y_test)
    # visualise predictions
    confusion_visual(true_labels, predicted_labels, epochs)
    # Visualise Errors 
    # plot_errors(true_labels, predicted_labels, epochs)

def task2_40000():

    # load the data set.
    X_train, y_train, X_test, y_test = generate_data(
        rows=500, cols=4, train_size=0.8)
    # Number of epochs the network will be trained on.
    epochs = 40000
    # Create a network.
    network = MultilayeredPerceptron(
        input_size=4, hidden_size=40, output_size=1, learning_rate=0.1, activation=sigmoid)
    # Train network.
    run_training(network, X_train, y_train, epochs)
    # Test Network Performance.
    run_testing(network, X_train, y_train, type='Train')
    run_testing(network, X_test, y_test)
    # get true and predicitons
    true_labels, predicted_labels = get_pred_values(network, X_test, y_test)
    # visualise predictions
    confusion_visual(true_labels, predicted_labels, epochs)
    # Visualise Errors 
    # plot_errors(true_labels, predicted_labels, epochs)

def test():
    pass


# ==================================================================== #
#                             MAIN                                     #
# ==================================================================== #


if __name__ == "__main__":
    task2_100()   # network = MultilayeredPerceptron(input_size=4, hidden_size=5, output_size=1, learning_rate=0.1, activation=sigmoid)
    #task2_1000()  # network = MultilayeredPerceptron(input_size=4, hidden_size=50, output_size=1, learning_rate=0.1, activation=sigmoid)
    # task2_10000() # network = MultilayeredPerceptron(input_size=4, hidden_size=500, output_size=1, learning_rate=0.1, activation=sigmoid)
    # task2_25000()
    # task2_40000()
    
    pass




#************************************************************************#

# ==================================================================== #
#                           END OF PROGRAM                             #
# ==================================================================== #

#************************************************************************#