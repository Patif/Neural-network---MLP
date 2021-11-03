from math import exp, log
import numpy as np


def sigmoid(x):
    return 1/(1 + exp(-x))


def ReLU(x):
    return 0 if x < 0 else x


def softplus(x):
    return log(1 + exp(x))


def softmax(z_vec):
    denominator = 0
    result = np.zeros(z_vec.shape)
    for i in range(z_vec.shape[0]):
        result[i, 0] = exp(z_vec[i, 0])
        denominator += result[i, 0]
    return result/denominator


def sigmoid_deriv(z_vec):
    result = np.zeros(z_vec.shape)
    for i, z in enumerate(z_vec[0]):
        result[0][i] = sigmoid(z) * (1 - sigmoid(z))
    return z_vec


def softplus_deriv(z_vec):
    result = np.zeros(z_vec.shape)
    for i, z in enumerate(z_vec[0]):
        result[0][i] = 1/(1 + exp(-z))
    return z_vec


def get_deriv(function):
    if function == sigmoid:
        return sigmoid_deriv
    if function == softplus:
        return softplus_deriv
