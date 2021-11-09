from mlp import MLP
from utils import sigmoid, softplus, softmax
import numpy as np
from keras.datasets import mnist

if __name__ == "__main__":
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    X_train = np.array(np.reshape(X_train, (X_train.shape[0], X_train.shape[1]*X_train.shape[2], 1)), dtype=float)
    X_test = np.array(np.reshape(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2], 1)), dtype=float)
    N = X_train.shape[1]
    L = 20
    M = 60
    M_prim = 10
    neural_structure = [(N, sigmoid),
                        (M, softmax),
                        [M_prim]]
    network = MLP(neural_structure)
    network.train(X_train, Y_train, 100, 2.5, X_test[0:9500], Y_test[0:9500])
    print(network.assessment(X_test[9500:], Y_test[9500:]))
"""
    network = MLP(neural_structure)
    network.learn([(np.array([[0.2, 0.1, 0.2, 0.4,0.2, 0.1, 0.2, 0.4, 0.2, 0.2 ]]), np.array([[0], [1], [0], [0]])),
                   (np.array([[0.1, 0.2, 0.5, 0.7,0.1, 0.2, 0.5, 0.7,0.0,0.9]]), np.array([[1], [0], [0], [0]]))], 1, 0.1, 0.2,
                  [(np.array([[0.2, 0.1, 0.2, 0.4, 0.2, 0.1, 0.2, 0.4, 0.2, 0.2]]), np.array([[0], [1], [0], [0]])),
                   (np.array([[0.1, 0.2, 0.5, 0.7, 0.1, 0.2, 0.5, 0.7, 0.0, 0.9]]), np.array([[1], [0], [0], [0]]))]
                  )
    
"""

