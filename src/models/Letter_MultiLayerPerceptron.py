# ==================================================================== #
#                         Libraries                                    #
# ==================================================================== #


import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt



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
        # self.weights2 = self.weights2.T
        # Transpose the weights2 array so that it has the correct dimensions
        h_output_err = np.dot(self.weights2.T, error)
        # Now tweak the network
        self.weights2 += self.learning_rate * \
            np.dot((error * output_val * (1.0 - output_val)), hidden_val.T)
        self.weights1 += self.learning_rate * \
            np.dot((h_output_err * hidden_val * (1.0 - hidden_val)), inputs.T)
        return error

    def train(self, inputs, labels):
        # Train the network for a number of epochs
        return self.backward(inputs, labels)

    def predict(self, inputs):
        # Use the trained network to make predictions on new dat

        _, output_val = self.forward(inputs)
        return output_val


# ==================================================================== #
#               Methods for running Training and Testing               #
# ==================================================================== #


# function to run training of model
def run_training(network, x, y, epochs, display_output=True, graph_errors=False):
    errors = []
    for epoch in range(epochs):
        for c, row in enumerate(x):
            values = row.split(',')
            inputs = (np.asfarray(values[1:]))
            # convert data to categorical values
            converted_vals = ord(values[0]) - 64

            targets = (np.asfarray(converted_vals, dtype=float))
            targets = np.zeros(26) + 0.01
            targets[converted_vals-1] = 0.99
            error=network.train(inputs, targets)
            errors.append(error)
            # output_error = network.train(inputs, train)
        if display_output:
            print(f"Epoch {epoch}: Training Occurring ... Average Error: {np.mean(error)}")
    print(f"Average Error over Entire Network: {np.mean(errors)}")
    if graph_errors:
        # Convert the array into a one-dimensional array
        errors = np.squeeze(errors)

        plt.figure()
        # Customize the chart title, x-axis label, and y-axis label
        plt.plot(errors)
        plt.xlabel('Training iteration')
        plt.ylabel('Error value')
        plt.title('MLP model error over time')
        plt.tight_layout()
        plt.show()
        # plt.savefig("graphing_errors_task2_{epochs}.png")


# function to run testing of model

def run_testing(network, x, y, type='Test', save_to_file=False):
    num_correct = []
    true_labels = []
    predicted_labels = []
    for count, row in enumerate(x):
        values = row.split(',')

        # convert data to categorical values
        true_l = ord(values[0]) - 64 - 1

        inputs = (np.asfarray(values[1:]))
        y_pred = network.predict(inputs)

        pred_l = np.argmax(y_pred)
        true_labels.append(true_l)
        predicted_labels.append(pred_l)

        if (pred_l) == (true_l):
            num_correct.append(1)
        else:
            num_correct.append(0)

    correct_array = np.asarray(num_correct)
    performance = correct_array.sum() / correct_array.size
    print(f"\nTesting {type} Performance = {performance}")
    print("Accuracy Score = ", accuracy_score(true_labels, predicted_labels))

    if save_to_file:
        with open(f"scores_task3_{count+1}_{type}.txt", "w") as f:
            f.write(f"\nTesting {type} Performance = {performance}")
        f.close()
    pass



# ==================================================================== #
#                   Download Data for Model                            #
# ==================================================================== #


def generate_data():
    # #load the data set
    with open("../data/letter-recognition.data", "r") as dataset:
        letter_data = dataset.readlines()
    dataset.close()

    X_train = letter_data[0:16000]
    y_train = letter_data[0:16000]

    X_test = letter_data[16000:20001]
    y_test = letter_data[16000:20001]

    return X_train, y_train, X_test, y_test


pass

# ==================================================================== #
#                       TESTING SUITIES                                #
# ==================================================================== #


def task3_1000():

    # load the data set.
    X_train, y_train, X_test, y_test = generate_data()
    # Number of epochs the network will be trained on.
    epochs = 1000
    # Create a network.
    network = MultilayeredPerceptron(
        input_size=16, hidden_size=100, output_size=26, learning_rate=0.1, activation=sigmoid)
    # Train network.
    run_training(network, X_train, y_train, epochs, graph_errors=False)
    # Test Network Performance.
    run_testing(network, X_train, y_train, type='Train')
    run_testing(network, X_test, y_test)


# ==================================================================== #
#                             MAIN                                     #
# ==================================================================== #


if __name__ == "__main__":
    task3_1000()   # network = MultilayeredPerceptron(input_size=16, hidden_size=100, output_size=26, learning_rate=0.1, activation=sigmoid)

    pass


#************************************************************************#

# ==================================================================== #
#                           END OF PROGRAM                             #
# ==================================================================== #

#************************************************************************#