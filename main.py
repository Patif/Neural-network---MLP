from mlp import MLP
from utils import sigmoid, softplus, softmax, tanh
import numpy as np
from keras.datasets import mnist

if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = np.array(np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2], 1)), dtype=float)
    X_test = np.array(np.reshape(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2], 1)), dtype=float)
    N = X_train.shape[1]
    #L = 20
    M = 60
    M_prim = 10
    reps = 10

    acc = 0
    neural_structure = [(N, softplus),
                        (M, softmax),
                        [M_prim]]
    for i in range(reps):
        network = MLP(neural_structure)
        network.train(X_train, Y_train, 100, 2.5, X_test[0:9500], Y_test[0:9500])
        acc_i = network.assessment(X_test[9500:], Y_test[9500:])
        print(i, acc_i)
        acc += acc_i
    print(acc / reps, "softplus")

    acc = 0
    neural_structure = [(N, tanh),
                        (M, softmax),
                        [M_prim]]
    for i in range(reps):
        network = MLP(neural_structure)
        network.train(X_train, Y_train, 100, 2.5, X_test[0:9500], Y_test[0:9500])
        acc_i = network.assessment(X_test[9500:], Y_test[9500:])
        print(i, acc_i)
        acc += acc_i
    print(acc / reps, "tanh")


    acc = 0
    neural_structure = [(N, sigmoid),
                        (M, softmax),
                        [M_prim]]
    for i in range(reps):
        network = MLP(neural_structure)
        network.train(X_train, Y_train, 100, 2.5, X_test[0:9500], Y_test[0:9500])
        acc_i = network.assessment(X_test[9500:], Y_test[9500:])
        print(i, acc_i)
        acc += acc_i
    print(acc / reps, "sigmoid")
