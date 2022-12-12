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

# ================
# Implementation
# ================


COMP 30230: Connectionist Computing

Assignment: Multi-Layered Perceptron from Scratch

Student Number: 19428806

Worth: 30%

Date of Completion: 12/12/22

\*\*\*Encouraged to use graphs and tables to report on training trends and results


Introduction:

In this report, we describe the experiments we ran with an MLP (multi-layer perceptron) model trained from scratch on XOR data, Numeric Vectors and Letter Recognition. The XOR data is a simple dataset that contains four input-output pairs, where the output is the exclusive-or (XOR) of the two inputs. The goal of the experiments was to evaluate the performance of the MLP model on the XOR data, and to gain insights into its ability to learn and generalize.

The Numeric vectors takes 500 vectors containing 4 components each. The value of each component should be a random number between -1 and 1. These will be your input vectors.

The Letter Recognition model, can predict a letter by training on an array attributes extracted from images of the letters.

Trained 3 models:

1. `	`Model trained on XOR
1. `	`Model trained on (500,4) inputs
1. `	`Trained Models to recognise letter.

To train and evaluate the MLP model, we used the following methods:

- `	`Algorithms: We used the standard backpropagation algorithm to train the MLP model, with cross-entropy loss. The backpropagation algorithm calculates the gradient of the loss function with respect to the weights of the model and updates the weights in the direction that minimizes the loss. Cross-entropy loss is a measure of how well the model can predict the correct outputs for the given inputs.

- `	`Hyperparameters: Learning rate and activation function stay consistent throughout with values of 0.1, and ‘sigmoid’.







1. `	`XOR model: 2 input layers with, 4 hidden layers, a single output unit, and a sigmoid activation function for both the hidden and output layers. The number of hidden units and the activation function are important hyperparameters that control the complexity and expressiveness of the MLP model.

1. `	`Numeric Vectors: Optimised to 4 input layers, 40 hidden neurons and I output neuron, activation function is sigmoid.

1. `	`Letter Recognition: Optimised at 16 input layers, 100 hidden neurons and 26 output neurons, one for each letter, activation function is sigmoid.



Description of Coding choices:

The code is a neural network implemented in Python. It includes a class called MultilayeredPerceptron, which represents the network. The class contains methods for initializing the network, performing forward and backward propagation, training the network, and making predictions with the trained network.

Libraries

At the start of the code, several libraries are imported.

- `	`numpy is a mathematical library that is used to perform operations on arrays, such as matrix multiplication.

- `	`sklearn is a machine learning library that is used to perform preprocessing on the data, such as scaling the data, and to evaluate the performance of the trained model. random is used to generate random numbers, and pandas is used to read in and manipulate data.

- `	`matplotlib.pyplot is used to generate visualizations of the data and the performance of the model.

INIT MEHTOD

In the \_\_init\_\_ method of the MultilayeredPerceptron class, the network's weights and biases are initialized. The weights are initialized randomly using a normal distribution with a mean of 0 and a standard deviation that is inversely proportional to the square root of the size of the layer it connects to. The biases are not initialized in this code.

The forward method performs a forward propagation step of the network. It takes in the inputs and passes them through the hidden layer, applying the specified activation function to the output of the hidden layer. The output of the hidden layer is then passed through the output layer, and the final output of the network is returned.

The backward method performs a backward propagation step of the network. It first computes the output of the network using the forward method, and then calculates the error at the output layer by comparing the predicted output to the true labels. The error is then backpropagated through the network, and the weights and biases are updated in order to reduce the error.

The train method trains the network for a number of epochs by calling the backward method. The predict method uses the trained network to make predictions on new data. It calls the forward method to compute the output of the network given the inputs, and returns the predicted labels.



Description of the Experiments:

1. `	`TRAINED A MODEL ON XOR DATA


Predicting XOR (Exclusive OR) using an MLP (Multi-Layer Perceptron) can be challenging because the XOR function is not linearly separable. This means that it is not possible to draw a straight line in the input space that can perfectly separate the XOR's two classes of outputs (i.e., 0 and 1).

Predicting XOR (Exclusive OR) using an MLP (Multi-Layer Perceptron) can be challenging because the XOR function is not linearly separable. This means that it is not possible to draw a straight line in the input space that can perfectly separate the XOR's two classes of outputs (i.e., 0 and 1).

To understand why this is the case, consider the following truth table for the XOR function:

x1	x2	XOR

0	0	0

0	1	1

1	0	1

1	1	0

As you can see, the output of the XOR function is 1 whenever the input values x1 and x2 are different and 0 when they are the same. Because this behaviour is not linearly separable, it cannot be learned by an MLP with a single hidden layer (which is equivalent to a linear classifier).

To overcome this limitation, an MLP with at least two hidden layers is typically used to predict XOR. The first hidden layer can learn to transform the input data into a new representation that is linearly separable, and the second hidden layer can then use this representation to predict the XOR output. Despite this, it is still challenging to train an MLP to predict XOR accurately, especially if the dataset is small or if the model is not properly tuned.

Testing Results of my Model


Epochs	Inputs Layers	Hidden Layers	Output Layers	Learning rate	Performance accuracy	Average Error

100	2	4	1	0.1	0.75	- 0.10998

1000	2	4	1	0.1	0.75	0.008845

2000	2	4	1	0.1	0.75	-0.009642

5000	2	4	1	0.1	1.0	-0.001732

10000	2	4	1	0.1	1.0	-0.002376


Visualising Error for optimised inputs (training)



The two graphs above show the outputted errors for models trained on 2000 and 10000 epochs alike, we can see the average error is optimised on the right-hand graph when the number of epochs exceeds 8000 for this task.


Analysis of results:

The findings indicate that the model is able to accurately lower its error over time as the number of epochs increases. When the number of epochs exceeds 8000, the model is able to achieve perfect results for this task. This suggests that the optimal number of epochs for this model in this task ranges between 8000 and 10000. The use of 2 input layers, 4 hidden layers, and 1 output layer, as well as a learning rate of 0.1, also appears to be effective in achieving good performance accuracy and low average error.



1. `	`Model trained on 500 input vectors

This task involves generating 500 vectors containing 4 components each. Each component should be a random number between -1 and 1. These will be the input vectors for a machine learning model. The corresponding output for each vector should be the sine of a combination of the input components. Specifically, the output will be the sine of the sum of the first and third components minus the second and fourth components.

Next, a multi-layer perceptron (MLP) with 4 inputs, at least 5 hidden units, and one output will be trained on 400 of the examples. The remaining 100 examples will be used for testing the trained model. The goal of the training is to learn a function that can accurately predict the sine of the combination of the input components based on the input vectors.


Testing Results of my model

\*\*(40 hidden layers Identified as best performing number for the complexity of this problem)


Epochs	Inputs Layers	Hidden Layers	Output Layers	Learning rate	Train accuracy	Test Accuracy	Average Error

100	4	40	1	0.1	0.52	0.52	0.03684587

1000	4	40	1	0.1	0.74	0.74	-0.0045114

2000	4	40	1	0.1	0.7075	0.74	-0.0802324

5000	4	40	1	0.1	0.8125	0.81	-0.0032292

10000	4	40	1	0.1	0.775	0.67	-0.0030819

Analysing the table below, we identified a strong pattern between the values of 1000 – 2000 Epochs. The average error score given for 1000 epochs was very low considering the low sample size compared to the errors given at 5000 and 10000 with their large, weighted average. This led me to test between the 1000,2000 range to identify optimum epochs.


Epochs	Inputs Layers	Hidden Layers	Output Layers	Learning rate	Train accuracy	Test Accuracy	Average Error

1200	4	40	1	0.1	0.6975	0.66	-0.09556526

1400	4	40	1	0.1	0.725	0.69	0.030503258

1500	4	40	1	0.1	0.815	0.79	-0.00097820	

1600	4	40	1	0.1	0.775	0.77	0.051148315

1800	4	40	1	0.1	0.7675	0.83	0.002127673







Visualising Error for optimised inputs (training)

- `	`Line Graph on 1500 (0.815) Epochs, and 5000 (0.8125) Epochs respectively.






Visualising Confusion Matrix for optimised inputs (Testing)

- `	`Confusion Matrix on 1800 (0.83) Epochs, 5000(0.81) Epochs respectively.




Analysis of results:

Based on the information provided, it appears that the model performs best overall when trained for 5000 epochs, achieving a score of 0.8125 on the training set and 0.81 on the test set. However, it is worth noting that the model also performs comparably when trained for 1500 epochs on the training set and tested on the test set at 1800 epochs. This suggests that the model may not necessarily require a large number of training epochs in order to achieve good performance, and that there may be an optimal range of training epochs that leads to the best performance. Further experimentation and analysis may be needed to determine the exact relationship between the number of training epochs and model performance. 

\3.	Model trained on input Letter Recognition

In this experiment, we trained a multi-layer perceptron (MLP) on a dataset of images of letters. The dataset was obtained from the UCI Machine Learning repository, and it consists of 17 attributes extracted from the images of the letters, along with the corresponding target label indicating the letter that is present in the image. We split the dataset into a training set containing approximately 4/5 of the records, and a testing set containing the rest.

The MLP had 17 inputs, one for each attribute in the dataset, and 26 outputs, one for each letter of the alphabet. The number of hidden units in the MLP was determined by the user. The MLP was trained for at least 1000 epochs, and after training, we evaluated its performance on the testing set. The results showed that the MLP was able to accurately classify the images of letters based on the attributes extracted from the images.


Testing Results of my model

\*\*\*Running on 50 epochs to find optimised hidden layer

Epochs	Inputs Layers	Hidden Layers	Output Layers	Learning rate	Train Accuracy	Test Accuracy	Average Error

50	17	20	26	0.1	0.586375	0.5745	0.007677291609782978

50	17	40	26	0.1	0.6176875	0.606	0.008667029895667793

50	17	60	26	0.1	0.632625	0.61875	0.008065876126333171

50	17	80	26	0.1	0.621375	0.60675	0.010211993127618111

50	17	100	26	0.1	0.6354375	0.628	0.00877423988085484

50	17	150	26	0.1	0.653625	0.6425	0.010470926379658888

50	12	225	26	0.1	0.5615625	0.5495	0.012048882380014947

50	17	300	26	0.1	0.618	0.5975	0.013102927365818858

The optimal number of hidden layers for this problem was found to be around 100. When the number of hidden layers rises above 100, the model sacrifices performance timing and average low error for a slight increase in accuracy.









Visualising Error for optimised inputs

- `	`Epochs 50, hidden layer 100



Epochs	Inputs Layers	Hidden Layers	Output Layers	Learning rate	Train Accuracy	Test Accuracy	Average Error

50	17	100	26	0.1	0.6085625	0.59325	0.00850716



- `	`Epochs 250, Hidden Layers 100



Epochs	Inputs Layers	Hidden Layers	Output Layers	Learning rate	Train Accuracy	Test Accuracy	Average Error

250	17	100	26	0.1	0.7078125	0.6985	0.01155883



- `	`Epochs 500, Hidden Layers 100



Epochs	Inputs Layers	Hidden Layers	Output Layers	Learning rate	Train Accuracy	Test Accuracy	Average Error

250	17	100	26	0.1	0.665375	0.6525	0.010971655



Running Model on 1000 Epochs using 100 hidden, let’s see how we perform!

Epochs	Inputs Layers	Hidden Layers	Output Layers	Learning rate	Train Accuracy	Test Accuracy	Average Error

1000	17	100	26	0.1	0.70125	0.6825	0.011205373








Analysis of results:

Based on the information provided, it appears that the Letter model performs best when trained with 100 hidden layers and 1000 epochs. However, when the number of epochs is increased beyond 1000, the model takes a very long time to run. This suggests that there is an optimal range of training epochs for this model, and that increasing the number of epochs beyond this range does not necessarily lead to better performance, but instead leads to longer run times. Additionally, it is noted that the model can potentially perform better with more than 100 hidden layers, but this comes at the expense of other aspects of performance, such as run time. This indicates that there may be a trade-off between the number of hidden layers and other aspects of model performance, and that the optimal number of hidden layers may depend on the specific task and data at hand. Further experimentation and analysis may be needed to fully understand the relationship between the number of hidden layers and model performance.

Notes on Letter Recognition

1. `	`One problem I encountered, was overflow, when training on this model. Although I didn’t have the necessary time to alleviate this problem, I investigated the issue to find a solution. One solution I found was using an alternative activation function such as ReLU (Rectified Linear Unit). This function does not have the same issue with large negative inputs, so overflow would not be a problem.



Future Work

The future work section of this report will focus on exploring ways to improve the performance of machine learning models using regularization techniques, dropout, and learning rate tuning.

One potential direction for future work is to conduct a systematic study comparing the performance of different regularization techniques on a range of different tasks and datasets. This could provide insight into which techniques are most effective in different scenarios and could help to identify best practices for regularization.

Another potential direction for future work is to explore the use of more advanced techniques for tuning the learning rate. For example, rather than using a fixed learning rate throughout training, it may be beneficial to use a learning rate schedule that gradually decreases the learning rate over time. This can help the model to converge to a better solution and can potentially improve performance.

Additionally, future work could focus on developing more effective methods for selecting the optimal values for the various hyperparameters used in regularization and learning rate tuning. This could include using techniques such as grid search or random search to explore the hyperparameter space and identify the values that lead to the best performance.

Overall, by investigating these and other potential avenues for future work, we can continue to improve the performance of machine learning models and advance the state of the art in this field.



Conclusion


In this report, we described the implementation and testing of a multi-layer perceptron (MLP) in a programming language of our choice. Our MLP was able to create a new MLP with any given number of inputs, outputs, and hidden units, initialize the weights of the MLP to small random values, predict the outputs corresponding to an input vector, and implement learning by backpropagation.

We tested our MLP by training it on the XOR function and on a dataset of random vectors with corresponding outputs generated by the sine of a combination of the input components. The results showed that our MLP was able to accurately learn the XOR function and predict the outputs of the random vectors.

Additionally, we trained our MLP on a dataset of images of letters and evaluated its performance on the testing set. The results showed that our MLP was able to accurately classify the images of letters based on the attributes extracted from the images.

In addition to the results described above, our analysis also uncovered some other findings. For example, we found that the number of training epochs and the number of hidden units have a significant impact on the performance of the MLP. Specifically, we observed that increasing the number of training epochs and the number of hidden units generally led to better performance, but at the expense of longer run times.

Overall, our implementation, testing and analysis of the MLP demonstrated its ability to accurately learn and predict the outputs of various datasets as well as providing insight into the factors that affect the performance of the MLP and can help to guide future efforts to improve its performance.



END OF REPORT ON MULTILAYERED PERCEPTRON.

