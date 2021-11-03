import numpy as np
from copy import deepcopy
from math import log
from utils import get_deriv, softmax


class MLP:

    def __init__(self, structure):
        self.__w = []
        self.__bias = []
        self.__layer_activ_functions = []
        for i, (layer_size, activ_function, w_normal, bias_normal) in enumerate(structure[0:-1]):
            self.__w.append(np.random.normal(w_normal[0], w_normal[1], (structure[i+1][0], layer_size)))
            self.__bias.append(np.random.normal(bias_normal[0], bias_normal[1], (structure[i+1][0], 1)))
            self.__layer_activ_functions.append(activ_function)

    def sets(self, w, bias):
        self.__w = w
        self.__bias = bias

    @property
    def w(self):
        return self.__w

    @property
    def bias(self):
        return self.__bias

    @property
    def activation_functions(self):
        return self.__layer_activ_functions

    def change_layer_population(self, layer, new_population, w_normal, bias_normal):
        if layer < 0 or layer >= len(self.__layer_activ_functions) or new_population < 1:
            return False
        if layer == len(self.__layer_activ_functions) - 1:
            self.__w[layer] = np.random.normal(w_normal[0], w_normal[1], (new_population, self.__w[layer].shape[1]))
            return True
        if 0 <= layer:
            self.__w[layer] = np.random.normal(w_normal[0], w_normal[1], (self.__w[layer].shape[0], new_population))
        if 0 < layer:
            self.__w[layer - 1] = np.random.normal(w_normal[0], w_normal[1], (new_population, self.__w[layer].shape[1]))
            self.__bias[layer] = np.random.normal(bias_normal[0], bias_normal[1], (new_population, 1))
        return True

    def predict(self, data_set):
        predictions = []
        w_copy = deepcopy(self.__w)
        bias_copy = deepcopy(self.__bias)
        for data, _ in data_set:
            for i, w_matrix in enumerate(self.__w):
                z = w_matrix@data.transpose() + self.__bias[i]
                for j in range(z.shape[0]):
                    z[j][0] = self.__layer_activ_functions[i](z[j][0])
                data = z.transpose()
            predictions.append(softmax(data.transpose()))
        return predictions

    def learn(self, learning_set, batch_size, alpha, epsilon):
        w_copy = deepcopy(self.__w)
        bias_copy = deepcopy(self.__bias)
        batches = len(learning_set) // batch_size
        costs = [None, None]
        for i in range(batches):
            w_delta = []
            bias_delta = []
            for m in range(len(self.__bias)):
                w_delta.append(np.zeros(self.__w[m].shape))
                bias_delta.append(np.zeros(self.__bias[m].shape))
            costs[1] = 0
            for data, labels in learning_set[i*batch_size : (i+1)*batch_size]:
                j = None
                data = data.transpose()
                activations = [data]
                z_matrices = [data]
                for j, w_matrix in enumerate(self.__w):
                    data = activations[-1]
                    z = w_matrix @ data + self.__bias[j]
                    a = np.zeros((z.shape[0], 1))
                    for k in range(z.shape[0]):
                        a[k, 0] = self.__layer_activ_functions[j](z[k, 0])
                    activations.append(a)
                    z_matrices.append(z)
                y_pred = softmax(activations[-1])
                print("Ostatnie wyjście: " + str(activations[-1]))
                print("Softmax: " + str(y_pred))
                for z in range(y_pred.shape[0]):
                    y_pred[z, 0] = -log(y_pred[z, 0])
                costs[1] += np.sum(labels * y_pred)/batch_size
                error = (softmax(activations[-1]) - labels) * activations[-1]
                activations.pop()
                z_matrices.pop()
                while activations:
                    bias_delta[j] += error
                    w_delta[j] += error @ activations.pop().transpose()
                    error = self.__w[j].transpose() @ error * get_deriv(self.__layer_activ_functions[j])(z_matrices.pop())
                    j -= 1
            for j in range(len(self.__bias)):
                self.__bias[j] -= alpha/batch_size * bias_delta[j]
                self.__w[j] -= alpha/batch_size * w_delta[j]
            if costs[0] and (costs[1] - costs[0]) ** 2 > epsilon:
                return
            costs[0] = costs[1]

