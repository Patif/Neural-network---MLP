import numpy as np
from utils import get_deriv, softmax, softmax_deriv, momentum


class MLP:

    def __init__(self, structure, w_range=0.01, bias_range=0.01):
        self.__w = []
        self.__bias = []
        self.__layer_activ_functions = []
        for i, (layer_size, activ_function) in enumerate(structure[0:-1]):
            self.__w.append(np.random.randn(structure[i + 1][0], layer_size) * np.sqrt(2 / layer_size))
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

    def train(self, x_train, y_train, batch_size, alpha, x_val, y_val, epochs=20, gamma=0.7, optimizer=0, beta_1=0.9,
              beta_2=0.95, adadelta_gamma=0.99):
        batches = len(x_val) // batch_size
        best_w = None
        best_bias = None
        highest_acc = None
        epoch = 0
        while epoch < epochs:
            delta_w_last, delta_bias_last, grads_w_last, grads_bias_last, m_w, v_w, m_bias, v_bias, \
            adadelta_d_w_last, adadelta_d_bias_last, adadelta_s_w_last, adadelta_s_bias_last = None, None, None, \
                                                                                               None, None, None, \
                                                                                               None, None, None, \
                                                                                               None, None, None
            t = 1
            for i in range(batches):
                indices = np.random.randint(x_train.shape[0], size=batch_size)
                delta_w_last, delta_bias_last, grads_w_last, grads_bias_last, m_w, v_w, m_bias, v_bias, \
                adadelta_d_w_last, adadelta_d_bias_last, adadelta_s_w_last, \
                adadelta_s_bias_last = self.learn(
                    x_train[indices], y_train[indices], batch_size,
                    alpha, gamma,
                    optimizer=optimizer, grads_bias_last=grads_bias_last, grads_w_last=grads_w_last,
                    delta_w_last=delta_w_last,
                    delta_bias_last=delta_bias_last, beta_1=beta_1,
                    beta_2=beta_2, m_w_last=m_w,
                    v_w_last=v_w, m_bias_last=m_bias,
                    v_bias_last=v_bias, t=t, adadelta_gamma=adadelta_gamma,
                    adadelta_d_w_last=adadelta_d_w_last, adadelta_d_bias_last=adadelta_d_bias_last,
                    adadelta_s_w_last=adadelta_s_w_last, adadelta_s_bias_last=adadelta_s_bias_last)
                t += 1
            acc = self.assessment(x_val, y_val)
            print("Celnosc(walidacyjny): {} Epoka: {}".format(acc, epoch + 1))
            if best_w is None or acc > highest_acc:
                best_w = self.__w
                best_bias = self.__bias
                highest_acc = acc
            epoch += 1
        self.__w = best_w
        self.__bias = best_bias

    # optimizer
    # 0 - nie uzywaj
    # 1 - momentum
    # 2 - momentum nesterowa
    # 3 - adagard
    # 4 - adadelta
    # 5 - adam
    def learn(self, x_train, y_train, batch_size, alpha, gamma, optimizer=0,
              delta_w_last=None, delta_bias_last=None, grads_w_last=None, grads_bias_last=None, beta_1=0.9,
              beta_2=0.999, m_bias_last=None, v_bias_last=None, m_w_last=None, v_w_last=None, t=1, adadelta_gamma=0.99,
              adadelta_d_w_last=None, adadelta_d_bias_last=None, adadelta_s_w_last=None, adadelta_s_bias_last=None):
        delta_w = []
        delta_bias = []
        grads_w = []
        grads_bias = []
        adadelta_d_w = []
        adadelta_d_bias = []
        adadelta_s_w = []
        adadelta_s_bias = []
        m_w = []
        v_w = []
        m_bias = []
        v_bias = []
        epsilon = 1e-5
        for m in range(len(self.__bias)):
            delta_w.append(np.zeros(self.__w[m].shape))
            delta_bias.append(np.zeros(self.__bias[m].shape))
            grads_w.append(np.zeros(self.__w[m].shape))
            grads_bias.append(np.zeros(self.__bias[m].shape))
            m_bias.append(np.zeros(self.__bias[m].shape))
            v_bias.append(np.zeros(self.__bias[m].shape))
            m_w.append(np.zeros(self.__w[m].shape))
            v_w.append(np.zeros(self.__w[m].shape))
            adadelta_s_w.append(np.zeros(self.__w[m].shape))
            adadelta_s_bias.append(np.zeros(self.__bias[m].shape))
            adadelta_d_w.append(np.zeros(self.__w[m].shape))
            adadelta_d_bias.append(np.zeros(self.__bias[m].shape))
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
                    if optimizer == 2 and delta_w_last is not None:
                        error = (self.__w[j] - delta_w_last[j]).transpose() @ error \
                                * get_deriv(self.__layer_activ_functions[j - 1])(z_matrices.pop())
                    else:
                        error = self.__w[j].transpose() @ error * get_deriv(self.__layer_activ_functions[j - 1])(
                            z_matrices.pop())
                j -= 1
        for j in range(len(self.__bias)):
            if optimizer == 4:
                adadelta_s_w[j] = adadelta_gamma * (0 if adadelta_s_w_last is None else adadelta_s_w_last[j]) \
                                  + (1 - adadelta_gamma) * (delta_w[j] ** 2)
                adadelta_s_bias[j] = adadelta_gamma * (0 if adadelta_s_bias_last is None
                                                       else adadelta_s_bias_last[j]) + (1 - adadelta_gamma) * (
                                                 delta_bias[j] ** 2)

                delta_w[j] *= np.sqrt((0 if adadelta_d_w_last is None else adadelta_d_w_last[j]) + epsilon) / np.sqrt(
                    adadelta_s_w[j] + epsilon)
                delta_bias[j] *= np.sqrt(
                    (0 if adadelta_d_bias_last is None else adadelta_d_bias_last[j]) + epsilon) / np.sqrt(
                    adadelta_s_bias[j] + epsilon)

                adadelta_d_w[j] = adadelta_gamma * (0 if adadelta_d_w_last is None else adadelta_d_w_last[j]) + \
                                  (1 - adadelta_gamma) * (delta_w[j] ** 2)
                adadelta_d_bias[j] = adadelta_gamma * (0 if adadelta_d_bias_last is None
                                                       else adadelta_d_bias_last[j]) + (1 - adadelta_gamma) * (
                                                 delta_bias[j] ** 2)
            elif optimizer == 5:
                m_bias[j] = beta_1 * (0 if m_bias_last is None else m_bias_last[j]) + (1 - beta_1) * delta_bias[j]
                v_bias[j] = beta_2 * (0 if v_bias_last is None else v_bias_last[j]) + (1 - beta_2) * (
                        delta_bias[j] ** 2).sum()

                m_w[j] = beta_1 * (0 if m_w_last is None else m_w_last[j]) + (1 - beta_1) * delta_w[j]
                v_w[j] = beta_2 * (0 if v_w_last is None else v_w_last[j]) + (1 - beta_2) * (delta_w[j] ** 2).sum()

                corrected_m_bias = m_bias[j] / (1 - beta_1 ** t)
                corrected_v_bias = v_bias[j] / (1 - beta_2 ** t)

                corrected_m_w = m_w[j] / (1 - beta_1 ** t)
                corrected_v_w = v_w[j] / (1 - beta_2 ** t)

                delta_bias[j] = alpha * corrected_m_bias / (np.sqrt(corrected_v_bias).sum() + epsilon)
                delta_w[j] = alpha * corrected_m_w / (np.sqrt(corrected_v_w).sum() + epsilon)
            else:
                grads_bias[j] = delta_bias[j]
                grads_w[j] = delta_w[j]
                alpha_w, alpha_bias = alpha, alpha

                if optimizer in [1, 2] and delta_w_last is not None:
                    momentum_w, momentum_bias = momentum(delta_w_last, delta_bias_last, gamma)
                    delta_bias[j] += momentum_bias[j]
                    delta_w[j] += momentum_w[j]

                if optimizer == 3:
                    grads_bias[j] = grads_bias[j] ** 2 + (0 if grads_bias_last is None else grads_bias_last[j])
                    grads_w[j] = grads_w[j] ** 2 + (0 if grads_w_last is None else grads_w_last[j])
                    alpha_bias /= (np.sqrt(grads_bias[j].sum()) + epsilon)
                    alpha_w /= (np.sqrt(grads_w[j].sum()) + epsilon)

                if optimizer in [0, 1, 2]:
                    alpha_bias /= batch_size
                    alpha_w /= batch_size

                delta_bias[j] = alpha_bias * delta_bias[j]
                delta_w[j] = alpha_w * delta_w[j]

            self.__bias[j] -= delta_bias[j]
            self.__w[j] -= delta_w[j]
        return delta_w, delta_bias, grads_w, grads_bias, m_w, v_w, m_bias, v_bias, adadelta_d_w, adadelta_d_bias, adadelta_s_w, adadelta_s_bias
