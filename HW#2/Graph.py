import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-2, 5, 0.01)
y = 5*x**4 - 22.4*x**3 + 15.85272*x**2 + 24.161472*x - 23.4824832

plt.figure()
plt.plot(x, y)
plt.grid()
plt.xlim(-1, 3)
plt.ylim(-30, -15)
plt.show()