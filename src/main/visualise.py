import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from itertools import product


def visualise_learning(output_data, epochs):
    # Create a DataFrame from the output data
    
    # Use the DataFrame's plot method to create a line chart
    plt.figure()

    # Customize the chart title, x-axis label, and y-axis label
    plt.plot(output_data, title="Error over time",
            xlabel="Epoch", ylabel="Output value")

    plt.show()
    plt.savefig("output_chart_XOR_{epochs}.png")
    pass


def confusion_visual(true_labels, predicted_labels, epochs):
    bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    true_discrete = np.digitize(true_labels, bins)
    pred_discrete = np.digitize(predicted_labels, bins)
    cm = confusion_matrix(true_discrete, pred_discrete)

    # Plotting confusion matrix
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar()

    # add labels to the plot to show the number of true positive, true negative, false positive, and false negative predictions.
    classes = bins
    # classes = ["Class " + str(number) for number in range(1, 27)]

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()

    plt.show()
    plt.savefig(f"confusion_matrix_{epochs}.png")
    print("Matrix saved as .png")

    pass

def plot_errors(true_labels, predicted_labels, epochs):
    # Create the line graph
    # bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    # true_discrete = np.digitize(true_labels, bins)
    # pred_discrete = np.digitize(predicted_labels, bins)
    # list_epochs = list(range(1, epochs+1))
    # error_values = np.subtract(true_discrete, pred_discrete)
    error_values = np.subtract(true_labels, predicted_labels)
    # plt.plot(list_epochs, error_values, label='Training Loss')
    plt.plot(error_values, label='Training Loss')

    # Add labels and title
    plt.xlabel('Training iteration')
    plt.ylabel('Error value')
    plt.title('MLP model error over time')

    # Show the graph
    plt.show()
    plt.savefig(f"errors_{epochs}.png")
    print("Line plot saved as .png")
