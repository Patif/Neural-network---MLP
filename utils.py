import numpy as np


def sigmoid(x):
    x = np.clip(x, -500, 500)
    return 1/(1 + np.exp(-x))


def ReLU(x):
    return 0 if x < 0 else x


def tanh(z_vec):
    z_vec = np.clip(z_vec, -500, 500)
    exp_plus = np.exp(z_vec)
    exp_minus = np.exp(-z_vec)
    return (exp_plus - exp_minus)/(exp_plus + exp_minus)


def tanh_deriv(z_vec):
    z_vec = np.clip(z_vec, -500, 500)
    return 1 - tanh(z_vec)**2

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
    return sigmoid(z_vec) * (1 - sigmoid(z_vec))


def softplus_deriv(z_vec):
    z_vec = np.clip(z_vec, -500, 500)
    return 1/(1 + np.exp(-z_vec))


def momentum(delta_w_last, delta_bias_last, gamma):
    for i in range(len(delta_w_last)):
        delta_w_last[i] = np.dot(delta_w_last[i], gamma)
        delta_bias_last[i] = np.dot(delta_bias_last[i], gamma)
    return delta_w_last, delta_bias_last


def get_deriv(function):
    if function == sigmoid:
        return sigmoid_deriv
    if function == softplus:
        return softplus_deriv
    if function == softmax:
        return softmax_deriv
    if function == tanh:
        return tanh_deriv
