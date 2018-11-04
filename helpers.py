import numpy as np
from proj1_helpers import *
from features import *

def standardize(x):
    """Standardize the given data"""
    means = x.mean(0)
    stds = x.std(0)
    return (x - means)/stds

def compute_accuracy(y, tx, w):
    """Compute the accuracy of a model"""
    y_pred = predict_labels(w, tx)
    return 1 - np.sum(np.abs(y - y_pred)) / (2* len(y_pred))

def compute_loss_mse(y, tx, w, lambda_=0):
    """Calculate the loss for MSE."""
    e = y - tx.dot(w)
    return 1/2*np.mean(e**2) + (lambda_ / 2) * np.dot(w, w)

def compute_gradient_mse(y, tx, w, lambda_=0):
    """Compute the gradient for MSE."""
    e = y - tx.dot(w)
    grad = -tx.T.dot(e) / len(e) + lambda_ * w
    return grad

def sigmoid(z):
    """Compute the sigmoid function of the input"""
    return 1. / (1. + np.exp(-z))

def compute_gradient_lg(y, tx, w, lambda_ = 0):
     """Calculate the loss using Logistic Regression"""
     z = np.dot(tx, w)
     h = sigmoid(z)
     gradient = np.dot(tx.T, (h - y)) + lambda_ * w
     return gradient

def compute_loss_lg(y, tx, w, lambda_ = 0):
    """Calculate the loss using Logistic Regression"""
    z = np.dot(tx, w)
    h = sigmoid(z)
    loss = np.sum(np.log(1 + h) - np.multiply(y, z)) + (lambda_ / 2) * np.dot(w, w)
    return loss

def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]

def clean(input_data, mean=False):

    #Replace -999 by most frequent value of column
    for i in range(input_data.shape[1]):
        current_col = input_data[:, i]

        if -999.0 in current_col:
            indices_to_change = (current_col == -999.0)
            if mean:
                curr_mean = np.mean(current_col[~indices_to_change])
                current_col[indices_to_change] = curr_mean
            else:
                (values,counts) = np.unique(current_col[~indices_to_change], return_counts=True)
                ind=np.argmax(counts)
                current_col[indices_to_change] = values[ind] if len(values) > 0 else 0

    return input_data

def preprocess(input_data):

    #Clean the data
    cleaned = clean(input_data, mean=True)

    #Standardize and add features (polynomials)
    standardized = standardize(cleaned)
    added_features = add_features(standardized)
    added_features = standardize(added_features)

    #Add a column of ones for the bias term
    tx = np.c_[np.ones(len(input_data)), added_features]

    return tx
