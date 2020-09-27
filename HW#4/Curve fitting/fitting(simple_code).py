import numpy as np
import matplotlib.pyplot as plt

x = np.array([-2.9, -2.1, -0.9, 1.1, 0.1, 1.9, 3.1, 4.0])
y = np.array([35.4, 19.7, 5.7, 2.1, 1.2, 8.7, 25.7, 41.5])

fit = np.polyfit(x, y, 2)
print(fit)

r = np.arange(-4, 5, 0.01)
fit2 = 3.16052477*r**2 - 2.36059821*r + 1.35828072

plt.scatter(x, y)
plt.plot(r, fit2)
plt.show()