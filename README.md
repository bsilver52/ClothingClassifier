# ClothingClassifier

**Code Authors:** Ben Silver, Liam Drew     

**Project Objective:** Build a classifier to map a given grayscale image to the type of clothing it visualizes

## Data Preparation
The goal was to train a model to accurately categorize clothing. Because the provided Fashion MNIST dataset
contains variations in the number of examples for each clothing type, model training aimed to minimize balanced
accuracy on the heldout data. This prevents significant model performance discrepancies between
class labels from skewing the results. After splitting the dataset into train / test datasets, class labels with
fewer examples were duplicated and concatenated to the original training set. While this increased the training
set size, various data augmentation strategies were implemented to provide unique pictures. Each image was reshaped
our images from 2D (n x 784) to 3D (n x 28 x 28) to be compatible with the numpy and scikit image augmentation
libraries. Functions were derived to flip images vertically /horizontally, rotate images by various angle values,
and shift images by various coordinate pairs. These functions were applied over the existing class labels with less
data, and the new images were concatenated to the training data.

## Hyperparameter Selection and Model Evaluation
Various neural networks and logistic regression models were implemented to derive an optimal classifier. Ultimately,
using a multi-layer perceptron (MLP) classifier yielded the best results. An Adam solver was used to perform
gradient descent and maximize balanced accuracy. Three hyperparameters were tuned: learning rate initialization,
which controls the model step sizes during gradient descent, batch size, which controls the number of examples used
to compute gradients for a given set of weights, and hidden layer size, which controls the number of neurons in each
hidden layer. A grid search with a five fold cross-validation was implemented using log-spaced learning rate values
in [0.0001, 10] and linearly spaced batch size and hidden layer values from [10,120] and [0,250], respectively. We 
also initialized different random states. A random search cross-validation (50 iterations) was then implemented using
normal and lognormal distributions centered around the initial hyperparameter values that maximized balanced accuracy.
The best model yielded balanced accuracy scores of  0.938 and 0.926 on the heldout and test datasets, respectively.
The low difference between the heldout and test set scores attests to the effectiveness of the model and its ability to
generalize well to new datasets.
