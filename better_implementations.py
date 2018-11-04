from helpers import *
import numpy as np

def accelerated_least_squares_GD(y, tx, initial_w, max_iters, lambda_=0):
    """Gradient descent algorithm."""
    #Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    #Parameters for AGD
    yw = w

    #Compute the eigenvalues to find find L and Mu
    eigenvalues = np.linalg.eigvals(tx.T.dot(tx))
    L = np.sqrt(np.amax(eigenvalues)+lambda_).astype('float64')
    mu = np.sqrt(np.amin(eigenvalues)+lambda_).astype('float64')

    for n_iter in range(max_iters):
        #Compute the gradient
        gradient = compute_gradient_mse(y, tx, yw, lambda_)

        #accelerated gradient descent
        w = yw - (2./(L+mu)) * gradient

        #Adaptive restart
        if gradient.T.dot(w -ws[-1]) > 0:
            ws[-1] = w

        #accelerated gradient descent
        yw = w + (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu)) * (w -ws[-1])

        #Compute the loss
        loss = compute_loss_mse(y, tx, w, lambda_)

        #Store w and loss
        ws.append(w)
        losses.append(loss)

        print("Gradient Descent({bi}/{ti}): loss={l}, gradient={gradient}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, gradient=gradient))
    print("Performed Least Squares GD with Gamma: ", 2/(L+mu), " Lambda: ", lambda_)

    return losses, w

def accelerated_reg_logistic_regression(y, tx, initial_w, max_iters, lambda_=0):
    """Stochastic gradient descent algorithm."""

    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    #Make the labels of y become 0 or 1 in order for the loss function to work

    #Parameters for AGD
    y = (y + 1) / 2
    yw = w

    #Compute the eigenvalues to find L and Mu
    eigenvalues = np.linalg.eigvals(tx.T.dot(tx) + 2*lambda_*np.identity(tx.shape[1]))
    L = (1./4)*np.amax(eigenvalues).astype('float64')
    mu = (1./4)*np.amin(eigenvalues).astype('float64')

    for n_iter in range(max_iters):
        #Compute the gradient
        gradient = compute_gradient_lg(y, tx, yw, lambda_)

        #accelerated gradient descent
        w = yw - (2./(L+mu)) * gradient

        #Adaptive restart
        if gradient.T.dot(w -ws[-1]) > 0:
            ws[-1] = w

        #accelerated gradient descent
        yw = w + (np.sqrt(L) - np.sqrt(mu)) / (np.sqrt(L) + np.sqrt(mu)) * (w -ws[-1])

        #Compute the loss
        loss = compute_loss_lg(y, tx, w, lambda_)

        #Store w and loss
        ws.append(w)
        losses.append(loss)

        print("Stochastic Gradient Descent({bi}/{ti}): loss={l}, gradient={gradient}".format(
          bi=n_iter, ti=max_iters - 1, l=loss, gradient=gradient))
    print("Performed Logistic SGD with Gamma: ", 2./(L+mu), " Lambda: ", lambda_)

    return losses, w
