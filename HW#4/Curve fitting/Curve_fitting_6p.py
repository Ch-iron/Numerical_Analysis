import random
import numpy as np
import matplotlib.pyplot as plt

D = np.array([[-2.9, 35.4], [-2.1, 19.7], [-0.9, 5.7], [1.1, 2.1],
              [0.1, 1.2], [1.9, 8.7], [3.1, 25.7], [4.0, 41.5]])

for j in range(1, 3):
    # Randomly select 6 points
    r = random.sample(range(0, 8), 6)
    R = np.empty((0, 2), dtype=float)
    for i in r:
        R = np.append(R, [D[i]], axis = 0)
    print(R)

    # Make B
    B = np.empty((0, 1), dtype=float)
    for i in range(0, 6):
        B = np.append(B, [[R[i, 1]]], axis = 0)
    ## Make 1 column of A
    A1 = np.empty((0, 1), dtype=float)
    for i in range(0, 6):
        A1 = np.append(A1, [[(R[i, 0])**2]], axis = 0)
    ## Make 2 column of A
    A2 = np.empty((0, 1), dtype=float)
    for i in range(0, 6):
        A2 = np.append(A2, [[R[i, 0]]], axis = 0)
    ## Make 3 column of A
    A3 = np.ones((6, 1), dtype=float)
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
    if j == 1:
        z1 = X[0, 0]
        y1 = X[1, 0]
        x1 = X[2, 0]
    elif j == 2:
        z2 = X[0, 0]
        y2 = X[1, 0]
        x2 = X[2, 0]

k = np.arange(-4, 5, 0.01)
fit1 = z1*k**2 + y1*k + x1
plt.plot(k, fit1)
fit2 = z2*k**2 + y2*k + x2
plt.plot(k, fit2)
fit3 = 3.16052477*k**2 - 2.36059821*k + 1.35828072
#plt.plot(k, fit3)
plt.show()