# MultiLayeredPerceptron
Implemented a MultiLayeredPerceptron from Scratch

This project is a multi-layer perceptron (MLP) built in Python without using any available neural network/connectionist/machine learning libraries. The software is able to create a new MLP with a specified number of inputs, outputs, and hidden units, initialize the weights, predict outputs for an input vector, and implement learning by backpropagation.

## What to do, minimally

- Create a new MLP with any given number of inputs, any number of outputs (can be sigmoidal or linear), and any number of hidden units (sigmoidal/tanh) in a single layer.
- Initialize the weights of the MLP to small random values.
- Predict the outputs corresponding to an input vector.
- Implement learning by backpropagation.

## Testing

To test the software, follow these steps:

1. Train an MLP with 2 inputs, 3-4+ hidden units, and one output on the following examples (XOR function):
   - ((0, 0), 0)
   - ((0, 1), 1)
   - ((1, 0), 1)
   - ((1, 1), 0)
2. At the end of training, check if the MLP predicts correctly all the examples.
3. Generate 500 vectors containing 4 components each. The value of each component should be a random number between -1 and 1. These will be your input vectors. The corresponding output for each vector should be the sin() of a combination of the components. Specifically, for inputs:
   - [x1 x2 x3 x4]
   the (single component) output should be:
   - sin(x1-x2+x3-x4)
4. Train an MLP with 4 inputs, at least 5 hidden units, and one output on 400 of these examples and keep the remaining 100 for testing.
5. Calculate the error on the training and test sets. Compare the errors and evaluate whether the MLP has learned satisfactorily.

