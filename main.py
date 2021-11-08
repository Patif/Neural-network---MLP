from mlp import MLP
from utils import sigmoid, softplus
import numpy as np
from keras.datasets import mnist

if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    y_train = np.zeros((10, len(Y_train)))
    for i, value in enumerate(Y_train):
        y_train[value, i] = 1
    y_test = np.zeros((10, len(Y_test)))
    for i, value in enumerate(Y_test):
        y_test[value, i] = 1
    X_train = np.array(np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2], 1))/255, dtype=float)
    X_test = np.array(np.reshape(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2], 1))/255, dtype=float)
    X_train = np.round(X_train, 3)
    X_test = np.round(X_test, 3)
    N = X_train.shape[1]
    L = 30
    M = 20
    M_prim = 10
    neural_structure = [(N, softplus, (0, 0.1), (0, 0.1)),
                        (L, sigmoid, (0, 0.1), (0, 0.1)),
                        (M, softplus, (0, 0.1), (0, 0.1)),
                        [M_prim]]
    network = MLP(neural_structure)
    network.learn(X_train, y_train, 100, 0.01, X_test, y_test)
    print(network.assessment(X_test[500:1000], y_test[:, 500:1000]))
"""
    network = MLP(neural_structure)
    network.learn([(np.array([[0.2, 0.1, 0.2, 0.4,0.2, 0.1, 0.2, 0.4, 0.2, 0.2 ]]), np.array([[0], [1], [0], [0]])),
                   (np.array([[0.1, 0.2, 0.5, 0.7,0.1, 0.2, 0.5, 0.7,0.0,0.9]]), np.array([[1], [0], [0], [0]]))], 1, 0.1, 0.2,
                  [(np.array([[0.2, 0.1, 0.2, 0.4, 0.2, 0.1, 0.2, 0.4, 0.2, 0.2]]), np.array([[0], [1], [0], [0]])),
                   (np.array([[0.1, 0.2, 0.5, 0.7, 0.1, 0.2, 0.5, 0.7, 0.0, 0.9]]), np.array([[1], [0], [0], [0]]))]
                  )
    
"""

