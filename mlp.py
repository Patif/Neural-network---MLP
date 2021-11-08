import numpy as np
from copy import deepcopy
from math import log
from utils import get_deriv, softmax, softmax_deriv
from random import randrange


class MLP:

    def __init__(self, structure, w_range=0.01, bias_range=0.01):
        self.__w = []
        self.__bias = []
        self.__layer_activ_functions = []
        for i, (layer_size, activ_function, w_normal, bias_normal) in enumerate(structure[0:-1]):
            self.__w.append(np.random.normal(w_normal[0], w_normal[1], (structure[i + 1][0], layer_size)))
            self.__bias.append(np.random.normal(bias_normal[0], bias_normal[1], (structure[i + 1][0], 1)))
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

    def assessment(self, x_test, y_test):
        correct_preds = 0
        for i, data in enumerate(x_test):
            for index, w_matrix in enumerate(self.__w):
                z = np.round(w_matrix @ data + self.__bias[index], 8)
                for j in range(z.shape[0]):
                    z[j][0] = self.__layer_activ_functions[index](z[j][0])
                data = z
            output_layer = softmax(data)
            prediction = None
            max_prob = None
            result = None
            for j, probability in enumerate(output_layer):
                if max_prob is None or probability > max_prob:
                    prediction = j
                    max_prob = probability
                if y_test[j, i] == 1:
                    result = j
            if result == prediction:
                correct_preds += 1
        return correct_preds / len(x_test)

    def learn(self, x_train, y_train, batch_size, alpha, x_val, y_val, epochs=30):
        batches = len(x_train) // batch_size
        best_w = None
        best_bias = None
        highest_acc = None
        precision = 6
        epoch = 0
        while epoch < epochs:
            w_delta = []
            bias_delta = []
            for m in range(len(self.__bias)):
                w_delta.append(np.zeros(self.__w[m].shape))
                bias_delta.append(np.zeros(self.__bias[m].shape))
            for i in range(batches):
                for index, data in enumerate(x_train[i * batch_size: (i + 1) * batch_size]):
                    j = None
                    activations = [np.round(data, precision)]
                    z_matrices = []
                    for j, w_matrix in enumerate(self.__w):
                        data = activations[-1]
                        z = np.round(w_matrix @ data + self.__bias[j], precision)
                        a = np.zeros((z.shape[0], 1), dtype=float)
                        for k in range(z.shape[0]):
                            a[k, 0] = self.__layer_activ_functions[j](z[k, 0])
                        activations.append(np.round(a, 8))
                        z_matrices.append(z)
                    y_pred = softmax(activations[-1])
                    for z in range(y_pred.shape[0]):
                        y_pred[z, 0] = -log(y_pred[z, 0])
                    error = np.round((softmax(activations.pop()) - y_train[:, i * batch_size + index].reshape(10, 1)) * softmax_deriv(z_matrices.pop()), precision)
                    while j >= 0:
                        bias_delta[j] += error
                        w_delta[j] += error @ activations.pop().transpose()
                        if z_matrices:
                            error = np.round(self.__w[j].transpose() @ error * get_deriv(self.__layer_activ_functions[j])(
                                z_matrices.pop()), precision)
                        j -= 1
                for j in range(len(self.__bias)):
                    self.__bias[j] -= np.round(alpha / batch_size * bias_delta[j], precision)
                    self.__w[j] -= np.round(alpha / batch_size * w_delta[j], precision)
            acc = self.assessment(x_val, y_val)
            print("Celnosc(walidacyjny): {} Epoka: {}".format(acc, epoch + 1))
            if best_w is None or acc > highest_acc:
                best_w = self.__w
                best_bias = self.__bias
                highest_acc = acc
            epoch += 1
        self.__w = best_w
        self.__bias = best_bias
