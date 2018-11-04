import numpy as np
from helpers import *
from implementations import *
from better_implementations import *

def build_k_indices(N, k_fold, seed):
    """build k indices for k-fold."""
    interval = int(N / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(N)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)


def cross_validation(y, x, k_indices, k, model, initial_w=None, max_iters=None, gamma=None, lambda_=None):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    te_indice = k_indices[k]
    tr_indice = k_indices[~(np.arange(k_indices.shape[0]) == k)]
    tr_indice = tr_indice.reshape(-1)

    #Split the data
    y_te = y[te_indice]
    y_tr = y[tr_indice]
    tx_te = x[te_indice]
    tx_tr = x[tr_indice]

    #Compute loss and optimal weights for the chosen model
    loss, optimal_weights = choose_model(y_tr, tx_tr, model, initial_w, max_iters, gamma, lambda_)

    # calculate the loss for train and test data
    loss_tr = np.sqrt(2 * compute_loss_mse(y_tr, tx_tr, optimal_weights))
    loss_te = np.sqrt(2 * compute_loss_mse(y_te, tx_te, optimal_weights))

    # calculate the accuracy for train and test data
    acc_tr = compute_accuracy(y_tr, tx_tr, optimal_weights)
    acc_te = compute_accuracy(y_te, tx_te, optimal_weights)
    return loss_tr, loss_te, acc_tr, acc_te

def choose_model(y, tx, model, initial_w=None, max_iters=None, gamma=None, lambda_=None):
    """Choose a method and evaluate the optimal weights (and loss)"""
    if model == "least_squares":
        return least_squares(y, tx)
    elif model == "least_squares_GD":
        return least_squares_GD(y, tx, initial_w, max_iters, gamma, lambda_)
    elif model == "least_squares_SGD":
        return least_squares_SGD(y, tx, initial_w, max_iters, gamma, lambda_)
    elif model == "ridge_regression":
        return ridge_regression(y, tx, lambda_)
    elif model == "logistic_regression":
        return logistic_regression(y, tx, initial_w, max_iters, gamma)
    elif model == "reg_logistic_regression":
        return reg_logistic_regression(y, tx, initial_w, max_iters, gamma, lambda_)
    elif model == "accelerated_least_squares_GD":
        return accelerated_least_squares_GD(y, tx, initial_w, max_iters, lambda_)
    elif model == "accelerated_reg_logistic_regression":
        return accelerated_reg_logistic_regression(y, tx, initial_w, max_iters, lambda_)
