# scattering 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 
import cv2

src = cv2.imread("origin/Image3_crop.jpg", cv2.IMREAD_COLOR)
dst = cv2.cvtColor(src, cv2.COLOR_BGR2Lab)
print(dst.shape)
z = dst.reshape(-1, 3)
print(z.shape)

fig = plt.figure()
ax = fig.gca(projection='3d')

x = dst[:,:,0]
y = dst[:,:,1]
z = dst[:,:,2]
ax.scatter(x, y, z)
ax.set_xlabel('L')
ax.set_ylabel('a')
ax.set_zlabel('b')
plt.suptitle("ax.scatter")
plt.show()