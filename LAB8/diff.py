import numpy as np

x = np.array([[1, 3, 5, 7, 9],
              [2, 4, 5, 8, 0]])

zeros = np.zeros((x.shape[0], 1), dtype=int)
x = np.concatenate((zeros, x), axis=1)
differences = np.diff(x)

print(differences)