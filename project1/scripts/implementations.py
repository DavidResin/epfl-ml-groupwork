# Useful imports
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from proj1_helpers import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """calculates the least squares solution using gradien descent."""
    # Define parameters to store w and loss
    w = initial_w

    for n_iter in range(max_iters):

        gradient, _ = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)

        w = w - gamma * gradient

    return w, loss

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
    """calculates the least squares solution using stochastic gradien descent."""
    # Define parameters to store w and loss
    w = initial_w
    loss = None

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):

            # compute a stochastic gradient and loss
            gradient, _ = compute_gradient(y_batch, tx_batch, w)

            # update w through the stochastic gradient update
            w = w - gamma * gradient

            # calculate loss
            loss = compute_mse(y, tx, w)

    return w, loss

def least_squares(y, tx):
    """calculates the least squares solution."""
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    loss = compute_mse(y, tx, w)

    return w, loss

def ridge_regression(y, tx, lambda_):
    """implement ridge regression."""
    lambdaI = 2 * tx.shape[0] * lambda_ * np.identity(tx.shape[1])
    A = (tx.T @ tx) + lambdaI
    B = tx.T @ y
    w = np.linalg.solve(A, B)
    loss = np.sqrt(2 * compute_mse(y, tx, w))

    return w, loss

def sigmoid(t):
    """apply sigmoid function on t."""
    return np.exp(t) / (1 + np.exp(t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood."""
    sigmo = sigmoid(tx.dot(w))
    vector_loss = y.T.dot(np.log(sigmo)) + (1 - y).T.dot(np.log(1 - sigmo))
    return np.squeeze(-vector_loss)

def calculate_gradient(y, tx, w):
    """compute the gradient of loss."""
    right = sigmoid(tx.dot(w))-y
    gradient = tx.T.dot(right)
    return gradient

def calculate_hessian(y, tx, w):
    """return the hessian of the loss function."""
    sig = sigmoid(tx.dot(w))
    s = np.multiply(sig, (1-sig))
    H = (tx.T * (s.T)).dot(tx)
    return H

def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """implements logistic regression using gradient descent or SGD."""

    threshold = 1e-8
    losses = []
    w = initial_w

    for iter in range(max_iters):

        grad = calculate_gradient(y, tx, w)
        hess = calculate_hessian(y, tx, w)
        w -= np.linalg.solve(hess, grad)

        loss = calculate_loss(y, tx, w)
        losses.append(loss)

        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, losses[-1]

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    threshold = 1e-8
    losses = []
    w = initial_w

    for iter in range(max_iters):

        loss = calculate_loss(y, tx, w) + lambda_ * (w.T @ w)
        grad = calculate_gradient(y, tx, w) + 2 * lambda_ * w #delete the 2*
        w -= gamma * grad

        losses.append(loss)
        
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, losses[-1]
