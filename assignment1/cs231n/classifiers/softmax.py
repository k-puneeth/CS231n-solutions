from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]
    for i in range(num_train):
      scores = X[i].dot(W)
      exp_score = np.exp(scores)
      correct_class_exponent = exp_score[y[i]]
      sum_of_exponents = np.sum(exp_score)
      softmax_score = correct_class_exponent/sum_of_exponents
      loss += -np.log(softmax_score)

      for j in range(num_classes):
        if(y[i]==j):
          dW[:, y[i]] += (exp_score[y[i]] / sum_of_exponents - 1) * X[i].T
        else:  
          dW[:, j] += (exp_score[j] / sum_of_exponents) * X[i].T
    
    loss /= num_train
    dW /= num_train

    dW += reg*2*W
    loss += reg*np.sum(np.power(W,2))
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    num_train = X.shape[0]
    scores = X.dot(W)
    exp_scores = np.exp(scores)
    correct_class_exponent = exp_scores[range(num_train), y]
    sum_of_exponents = np.sum(exp_scores, axis=1)
    softmax_scores = correct_class_exponent/sum_of_exponents
    loss += -np.sum(np.log(softmax_scores))

    loss /= num_train

    loss += reg*np.sum(np.power(W, 2))
    
    dscores = exp_scores / sum_of_exponents.reshape(-1, 1)

    dscores[range(num_train), y] -= 1

    dW = X.T.dot(dscores)

    dW /= num_train

    dW += 2*reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
