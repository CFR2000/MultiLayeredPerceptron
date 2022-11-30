from typing import Self
import numpy as np

class NeuralNetwork():
    def __init__ (self, input_layers, hidden_layers, output_layers, learning_rate):
        self.input_layers = input_layers
        self.hidden_layers = hidden_layers
        self.output_layers = output_layers
        self.learning_rate = learning_rate
        # self.epochs = epochs
        # self.bias_hidden = bias_hidden
        # self.bias_output = bias_ouput
        

        #link weight matrices, wih and who
        #weights inside the arrays are w_i_j, where link is from node i to node j in the next layer
        #w11 w21
        #w12 w22 etc
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

    def __str__(self):
        return f"Multi-layered Perceptron with: \n{self.input_layers} Input neurons,\n{self.hidden_layers} Hidden layers,\n{self.output_layers} Output neurons,\n{self.learning_rate} as the learning rate."

    #Activation function
    def sig(x, derive=False):
        if derive:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    # matrix multiplication
    def product(self, x,y):
        return np.dot(x,y)

    def matrix_2d(self, x):
        return np.array(x, ndmin=2).T

    def adjust_weights_bp(self, error, f_output, b_output, weight):
        #f_output = last output value
        #b_output = output value before last
        weight += self.learning_rate * self.product( self.product( (error * f_output), (1.0 - f_output) ) , b_output.T)
        return weight

#A Neural Network requires 3 functionalites [training, feedforward, backpropagate]

    #Function to train the network.
    def train(self, seq_inputs, seq_targets):

    #"""Feed Forward"""    
        #list of input values for the network
        inputs = self.matrix_2d(seq_inputs)

        #Supervised Learning so we have a list of the expected outputs we want.
        targets = self.matrix_2d(seq_targets)

        #calculatingthe hidden layer values/weighted sum, input matrix * weights_input_hidden
        h_in = self.product(inputs, self.wih)

        #passing hidden inputs into the activation function.
        h_out = self.sig(h_in)

        #next we want to calculate the weights of the values going into the output layer
        y_in = self.product(h_out, self.who)
        #getting value for the output layer, by passing the weighted sum to the activation.
        y_out = self.sig(y_in)

        #Now we find the difference between the actual output and the expected.
        out_error = targets - y_out

        #Now it's time to backpropagate through our network to adjust the weights by using the error to improve outputted values.
        h_out_err = self.product(out_error, self.who.T) # we transpose the weights of the hidden output bc we are working backwards

        #Now we use the hidden ouput layer to tweak our network and adjust the weights
        #backpropagating to adjust the weights
        self.who = self.adjust_weights_bp(out_error, y_out, h_out, self.who)
        self.whi = self.adjust_weights_bp(h_out_err, h_out, inputs, self.whi)


    #function to identify the output of the network
    def query(self, seq_inputs):
        inputs = self.matrix_2d(seq_inputs)
        #hidden layer values
        h_in = self.product(inputs, self.wih)
        #passing hidden inputs into the activation function.
        h_out = self.sig(h_in)
        #output layer
        y_in = self.product(h_out, self.who)
        #weighted sum to the activation.
        y_out = self.sig(y_in)
        return y_out