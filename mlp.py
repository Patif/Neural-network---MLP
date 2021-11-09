import numpy as np
from utils import get_deriv, softmax, softmax_deriv


class MLP:

    def __init__(self, structure, w_range=0.01, bias_range=0.01):
        self.__w = []
        self.__bias = []
        self.__layer_activ_functions = []
        for i, (layer_size, activ_function) in enumerate(structure[0:-1]):
            self.__w.append(np.random.randn(structure[i + 1][0], layer_size)*np.sqrt(2/layer_size))
            self.__bias.append(np.zeros((structure[i + 1][0], 1)))
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
                z = w_matrix @ data + self.__bias[index]
                data = self.__layer_activ_functions[index](z)
            prediction = np.argmax(data)
            if y_test[i] == prediction:
                correct_preds += 1
        return correct_preds / len(x_test)

    def train(self, x_train, y_train, batch_size, alpha, x_val, y_val, epochs=30):
        batches = len(x_val) // batch_size
        best_w = None
        best_bias = None
        highest_acc = None
        epoch = 0
        while epoch < epochs:
            for i in range(batches):
                indices = np.random.randint(x_train.shape[0], size=batch_size)
                self.learn(x_train[indices], y_train[indices], batch_size, alpha)
            acc = self.assessment(x_val, y_val)
            print("Celnosc(walidacyjny): {} Epoka: {}".format(acc, epoch + 1))
            if best_w is None or acc > highest_acc:
                best_w = self.__w
                best_bias = self.__bias
                highest_acc = acc
            epoch += 1
        self.__w = best_w
        self.__bias = best_bias

    def learn(self, x_train, y_train, batch_size, alpha):
        delta_w = []
        delta_bias = []
        for m in range(len(self.__bias)):
            delta_w.append(np.zeros(self.__w[m].shape))
            delta_bias.append(np.zeros(self.__bias[m].shape))
        for index, data in enumerate(x_train):
            j = None
            activations = [data]
            z_matrices = []
            for j, w_matrix in enumerate(self.__w):
                z = w_matrix @ activations[-1] + self.__bias[j]
                a = self.__layer_activ_functions[j](z)
                activations.append(a)
                z_matrices.append(z)
            y_1h = np.bincount([y_train[index]], minlength=self.__bias[-1].shape[0]).reshape(10, 1)
            error = (activations.pop() - y_1h) * get_deriv(self.__layer_activ_functions[-1])(z_matrices.pop())
            while j >= 0:
                delta_bias[j] += error
                delta_w[j] += error @ activations.pop().transpose()
                if z_matrices:
                    error = self.__w[j].transpose() @ error * get_deriv(self.__layer_activ_functions[j - 1])(
                        z_matrices.pop())
                j -= 1
        for j in range(len(self.__bias)):
            self.__bias[j] -= (alpha / batch_size) * delta_bias[j]
            self.__w[j] -= (alpha / batch_size) * delta_w[j]
