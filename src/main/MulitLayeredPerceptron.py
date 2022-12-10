import numpy as np

def sigmoid(x):
    # Define the sigmoid activation function
    return 1 / (1+np.exp(-x))

