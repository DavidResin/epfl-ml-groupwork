# Useful imports
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from utility import *

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
  """calculates the least squares solution using gradien descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        
        gradient = compute_gradient(y, tx, w)
        loss = compute_mse(y, tx, w)
        
        w = w - gamma*gradient
        
        # store w and loss
        ws.append(w)
        losses.append(loss)

    return losses, ws

def least_squares_SGD(y, tx, initial_w, batch_size, max_iters, gamma):
  """calculates the least squares solution using stochastic gradien descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        for y_batch, tx_batch in batch_iter(y, tx, batch_size=batch_size, num_batches=1):
            # compute a stochastic gradient and loss
            gradient, _ = compute_stoch_gradient(y_batch, tx_batch, w)
            # update w through the stochastic gradient update
            w = w - gamma*gradient
            # calculate loss
            loss = compute_mse(y, tx, w)
            # store w and loss
            ws.append(w)
            losses.append(loss)
            
    return losses, ws
  
def least_squares(y, tx):
  """calculates the least squares solution."""
    w = np.linalg.solve(tx.T@tx, tx.T@y)
    loss = compute_mse(y, tx, w)
    
    return loss, w
  #raise NotImplementedError

def ridge_regression(y, tx, lambda_):
  """implements ridge regression."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
  raise NotImplementedError
  
def logistic_regression(y, tx, initial_w, max_iters, gamma):
  """implements logistic regression using gradient descent or SGD."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
  raise NotImplementedError
  
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
  """implements regularized logistic regression using gradient descent or SGD."""
    # ***************************************************
    # INSERT YOUR CODE HERE
    # ridge regression: TODO
    # ***************************************************
  raise NotImplementedError