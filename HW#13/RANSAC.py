import numpy as np
import math
import matplotlib.pyplot as plt

def least_square(domain):
    # Make B
    B = np.zeros((domain.shape[0], 1), dtype=np.float32)
    for i in range(0, domain.shape[0]):
        B[i] = domain[i, 1]

    ## Make A
    A = np.zeros((domain.shape[0], 2), dtype=np.float32)
    for i in range(0, domain.shape[0]):
        A[i, 0] = domain[i, 0]
        A[i, 1] = 1

    # pseudo-inverse
    At = np.transpose(A)
    AtA = np.dot(At, A)
    AtAI = np.linalg.inv(AtA)
    PI = np.dot(AtAI, At)

    # X
    X = np.dot(PI, B)
    return X

x = np.array([-5.0, -4.0, -3.0, -2.0, -1.0, 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

np.random.seed(10)
## original equation
noise = np.random.normal(0, math.sqrt(2), 12)
y = (2 * x - 1) + noise
print(y)

## Estimate with 12 points
original = np.zeros((12, 2), dtype = np.float32)
for i in range(0, 12):
    original[i, 0] = x[i]
    original[i, 1] = y[i]

X = least_square(original)
print("Original : y = " + str(X[0]) + "x + " + str(X[1]), end=" ")
error = 0
for i in range(0, 12):
    error = error + ((X[0] * original[i, 0] + X[1]) - original[i, 1])**2
print("Error : " + str(error))
print()

##Estimate with 6 points by random sampling
min_error = 100.0

sampling = np.zeros((6, 2), dtype = np.float32)
for j in range(0, 462):
    rand = np.random.choice(12, 6, replace=False)

    for i in range(0, 6):
        sampling[i, 0] = x[rand[i]]
        sampling[i, 1] = y[rand[i]]

    X = least_square(sampling)

    error = 0
    for i in range(0, 6):
        error = error + ((X[0] * sampling[i, 0] + X[1]) - sampling[i, 1])**2

    if error < min_error:
        min_error = error
        min_X = X
        min_sample = rand

print(np.sort(rand) + 1, end=" ")
print("y = " + str(min_X[0]) + "x + " + str(min_X[1]), end=" ")
print("Error : " + str(min_error))

plt.scatter(x, y)
origin_equation = 2 * x - 1
plt.plot(x, origin_equation)
best_sol = min_X[0] * x + min_X[1]
plt.plot(x, best_sol)
plt.grid()
plt.show()
