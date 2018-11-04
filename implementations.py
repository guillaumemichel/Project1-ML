from helpers import *
import numpy as np

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm for the Least Squares loss function"""
    # Define parameters to store w and loss
    losses = []
    w = initial_w

    for n_iter in range(max_iters):

        #Compute the gradient
        gradient = compute_gradient_mse(y, tx, w)

        #Update the weight vector
        w = w - gamma * gradient

        #Compute the loss
        loss = compute_loss_mse(y, tx, w)

        #Store loss
        losses.append(loss)

        print("Gradient Descent({bi}/{ti}): loss={l}, gradient={gradient}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, gradient=gradient))
    print("Performed Least Squares GD with Gamma: ", gamma)
    return losses, w

def ridge_regression(y, tx, lambda_=0):
    """Computes optimal weights using Ridge Regression with matrix operations"""
    #Store the dimensions of training data matrix
    N, d = tx.shape

    #Matrix operations for ridge regression
    a = tx.T.dot(tx) + 2 * N * lambda_ * np.identity(d)
    b = tx.T.dot(y)

    #Compute the optimal weights
    #To avoid the error of the singular matrix we "pseudo-solve" the system
    #using linalg.lstsq but if the matrix is not singular it is possible
    #to use linalg.solve
    
    optimal_weights =  np.linalg.lstsq(a, b)[0]

    #numpy.linalg.solve(a, b)
    
    #Compute the loss
    loss = compute_loss_mse(y, tx, optimal_weights)

    return loss, optimal_weights

def least_squares(y, tx):
    """Computes optimal weights using LS with matrix operations"""
    #Least Square is Ridge Regression with lambda = 0
    return ridge_regression(y, tx)

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic gradient descent algorithm for the Least Squares loss function"""

    #Define parameters to store w and loss
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        #For SGD, the batch has size 1
        for y_batch, tx_batch in batch_iter(y, tx, 1):

            #Compute the gradient
            gradient = compute_gradient_mse(y_batch, tx_batch, w)

            #Update the weight vector
            w = w - gamma * gradient

            #Compute the loss
            loss = compute_loss_mse(y, tx, w)

            #Store loss
            losses.append(loss)

        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, gradient={gradient}".format(
          bi=n_iter, ti=max_iters - 1, l=loss, gradient=gradient))

    print("Performed Least Squares SGD with Gamma: ", gamma)
    return losses, w


def reg_logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_=0):
    """Gradient descent algorithm for Logistic Regression with L2-regularization."""

    #Define parameters to store w and loss
    losses = []
    w = initial_w

    #Make the labels of y become 0 or 1 in order for the loss function to work
    y = (y + 1) / 2

    for n_iter in range(max_iters):
        #Compute the gradient
        gradient = compute_gradient_lg(y, tx, w, lambda_)

        #Update the weight vector
        w = w - gamma * gradient

        #Compute the loss
        loss = compute_loss_lg(y, tx, w, lambda_)

        #store w and loss
        losses.append(loss)

        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, gradient={gradient}".format(
          bi=n_iter, ti=max_iters - 1, l=loss, gradient=gradient))
    print("Performed Logistic SGD with Gamma: ", gamma, " Lambda: ", lambda_)
    return losses, w

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm for Logistic Regression."""
    return reg_logistic_regression(y, tx, initial_w, max_iters, gamma)
