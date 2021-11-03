from mlp import MLP
from utils import sigmoid, softplus
import numpy as np

if __name__ == "__main__":
    neural_structure = [(2, sigmoid, (-0.1, 0.1), (-1, 1)),
                        (3, sigmoid, (-0.1, 0.1), (-1, 1)),
                        [2]]
    network = MLP(neural_structure)
    network.sets([np.array([[0, 1], [2, 1], [2, 3]], dtype=float),
                  np.array([[0, 1, 0], [1, 2, 3]], dtype=float)],
                 [np.array([[1], [1], [1]], dtype=float),
                  np.array([[2], [0]], dtype=float)])
    network.learn([(np.array([[0.2, 0.1]]), np.array([[0], [1]]))], 1, 0.1, 0.2)

    N = 10
    L = 5
    M = 9
    M_prim = 4

    neural_structure = [(N, softplus, (-0.1, 0.1), (-1, 1)),
                        (L, sigmoid, (-0.1, 0.1), (-1, 1)),
                        (M, softplus, (-0.1, 0.1), (-1, 1)),
                        [M_prim]]

    network = MLP(neural_structure)
    network.learn([(np.array([[0.2, 0.1, 0.2, 0.4,0.2, 0.1, 0.2, 0.4, 0.2, 0.2 ]]), np.array([[0], [1], [0], [0]])),
                   (np.array([[0.1, 0.2, 0.5, 0.7,0.1, 0.2, 0.5, 0.7,0.0,0.9]]), np.array([[1], [0], [0], [0]]))], 1, 0.1, 0.2)
    


