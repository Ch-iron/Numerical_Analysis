import random
import numpy as np

D = np.array([[-2.9, 35.4], [-2.1, 19.7], [-0.9, 5.7], [1.1, 2.1],
              [0.1, 1.2], [1.9, 8.7], [3.1, 25.7], [4.0, 41.5]])

# Make B
B = np.empty((0, 1), dtype=float)
for i in range(0, 8):
    B = np.append(B, [[D[i, 1]]], axis = 0)
## Make 1 column of A
A1 = np.empty((0, 1), dtype=float)
for i in range(0, 8):
    A1 = np.append(A1, [[(D[i, 0])**2]], axis = 0)
## Make 2 column of A
A2 = np.empty((0, 1), dtype=float)
for i in range(0, 8):
    A2 = np.append(A2, [[D[i, 0]]], axis = 0)
## Make 3 column of A
A3 = np.ones((8, 1), dtype=float)
## Make A
A = np.hstack([A1, A2])
A = np.hstack([A, A3])

# pseudo-inverse
At = np.transpose(A)
AtA = np.dot(At, A)
AtAI = np.linalg.inv(AtA)
PI = np.dot(AtAI, At)

# X
X = np.dot(PI, B)
print(X)