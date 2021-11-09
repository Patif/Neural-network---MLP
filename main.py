from mlp import MLP
from utils import sigmoid, softplus, softmax
import numpy as np
from keras.datasets import mnist

if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = np.array(np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2], 1)), dtype=float)
    X_test = np.array(np.reshape(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2], 1)), dtype=float)
    X_train = np.round(X_train, 3)
    X_test = np.round(X_test, 3)
    N = X_train.shape[1]
    L = 30
    M = 20
    M_prim = 10
    neural_structure = [(N, sigmoid, (0.75, 1), (0.75, 1)),
                        (L, sigmoid, (0.75, 1), (0.75, 1)),
                        (M, softmax, (0.75, 1), (0.75, 1)),
                        [M_prim]]
    network = MLP(neural_structure)
    network.train(X_train, Y_train, 200, 2.5, X_test[0:2000], Y_test[0:2000])
    print(network.assessment(X_test, Y_test))
"""
    network = MLP(neural_structure)
    network.learn([(np.array([[0.2, 0.1, 0.2, 0.4,0.2, 0.1, 0.2, 0.4, 0.2, 0.2 ]]), np.array([[0], [1], [0], [0]])),
                   (np.array([[0.1, 0.2, 0.5, 0.7,0.1, 0.2, 0.5, 0.7,0.0,0.9]]), np.array([[1], [0], [0], [0]]))], 1, 0.1, 0.2,
                  [(np.array([[0.2, 0.1, 0.2, 0.4, 0.2, 0.1, 0.2, 0.4, 0.2, 0.2]]), np.array([[0], [1], [0], [0]])),
                   (np.array([[0.1, 0.2, 0.5, 0.7, 0.1, 0.2, 0.5, 0.7, 0.0, 0.9]]), np.array([[1], [0], [0], [0]]))]
                  )
    
"""

