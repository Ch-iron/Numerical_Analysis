import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-2, 5, 0.01)
y = 5*x**4 - 22.4*x**3 + 15.85272*x**2 + 24.161472*x - 23.4824832

plt.figure()
plt.plot(x, y)
plt.grid()
plt.xlim(3.1235, 3.1245)
plt.ylim(-0.05, 0.05)
plt.show()