import numpy as np


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1/(1 + np.exp(-x))


def ReLU(x):
    return 0 if x < 0 else x


def softplus(x):
    x = np.clip(x, -500, 500)
    return np.log(1 + np.exp(x))


def softmax(z_vec):
    z_vec = np.clip(z_vec, -500, 500)
    exps = np.exp(z_vec - z_vec.max())
    return exps / np.sum(exps)


def softmax_deriv(z_vec):
    return softmax(z_vec) * (1 - softmax(z_vec))


def sigmoid_deriv(z_vec):
    z_vec = np.clip(z_vec, -500, 500)
    result = np.zeros(z_vec.shape)
    for i, z in enumerate(z_vec[:, 0]):
        result[i, 0] = sigmoid(z) * (1 - sigmoid(z))
    return z_vec


def softplus_deriv(z_vec):
    z_vec = np.clip(z_vec, -500, 500)
    result = np.zeros(z_vec.shape)
    for i, z in enumerate(z_vec[:, 0]):
        result[i, 0] = 1/(1 + np.exp(-z))
    return z_vec


def get_deriv(function):
    if function == sigmoid:
        return sigmoid_deriv
    if function == softplus:
        return softplus_deriv
    if function == softmax:
        return softmax_deriv
