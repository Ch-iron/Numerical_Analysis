import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans

##np.set_printoptions(threshold=np.inf)
##np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=4)


X1 = np.random.normal(0, 3, 300)
Y1 = np.random.normal(0, 3, 300)
Z1 = np.random.normal(0, 3, 300)

X2 = np.random.normal(4, 3, 300)
Y2 = np.random.normal(4, 3, 300)
Z2 = np.random.normal(4, 3, 300)

X3 = np.random.normal(8, 3, 300)
Y3 = np.random.normal(8, 3, 300)
Z3 = np.random.normal(8, 3, 300)

X4 = np.random.normal(12, 3, 300)
Y4 = np.random.normal(12, 3, 300)
Z4 = np.random.normal(12, 3, 300)

X5 = np.random.normal(16, 3, 300)
Y5 = np.random.normal(16, 3, 300)
Z5 = np.random.normal(16, 3, 300)

X = np.hstack((np.hstack((np.hstack((np.hstack((X1, X2)), X3)), X4)), X5))
Y = np.hstack((np.hstack((np.hstack((np.hstack((Y1, Y2)), Y3)), Y4)), Y5))
Z = np.hstack((np.hstack((np.hstack((np.hstack((Z1, Z2)), Z3)), Z4)), Z5))
data = np.vstack((np.vstack((X, Y)), Z))
data = np.transpose(data)

fig = plt.figure()
ax = fig.gca(projection='3d')
fig2 = plt.figure()
ax2 = fig2.gca(projection='3d')

ax.scatter(X1, Y1, Z1, label="1")
ax.scatter(X2, Y2, Z2, label="2")
ax.scatter(X3, Y3, Z3, label="3")
ax.scatter(X4, Y4, Z4, label="4")
ax.scatter(X5, Y5, Z5, label="5")
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
fig.suptitle("Cluster_before")

kmeans = KMeans(n_clusters=5)
cluster_data = kmeans.fit(data)
print(kmeans.cluster_centers_)

K1 = [0, 0, 0]
K2 = [0, 0, 0]
K3 = [0, 0, 0]
K4 = [0, 0, 0]
K5 = [0, 0, 0]
for i in range(0, kmeans.labels_.size):
    if kmeans.labels_[i] == 0:
        K1 = np.vstack((K1, data[i]))
    elif kmeans.labels_[i] == 1:
        K2 = np.vstack((K2, data[i]))
    elif kmeans.labels_[i] == 2:
        K3 = np.vstack((K3, data[i]))
    elif kmeans.labels_[i] == 3:
        K4 = np.vstack((K4, data[i]))
    elif kmeans.labels_[i] == 4:
        K5 = np.vstack((K5, data[i]))

K1 = K1[1:K1.size]
K2 = K2[1:K2.size]
K3 = K3[1:K3.size]
K4 = K4[1:K4.size]
K5 = K5[1:K5.size]

KX1 = K1[:,0]
KY1 = K1[:,1]
KZ1 = K1[:,2]
KX2 = K2[:,0]
KY2 = K2[:,1]
KZ2 = K2[:,2]
KX3 = K3[:,0]
KY3 = K3[:,1]
KZ3 = K3[:,2]
KX4 = K4[:,0]
KY4 = K4[:,1]
KZ4 = K4[:,2]
KX5 = K5[:,0]
KY5 = K5[:,1]
KZ5 = K5[:,2]

ax2.scatter(KX1, KY1, KZ1, label="1")
ax2.scatter(KX2, KY2, KZ2, label="2")
ax2.scatter(KX3, KY3, KZ3, label="3")
ax2.scatter(KX4, KY4, KZ4, label="4")
ax2.scatter(KX5, KY5, KZ5, label="5")
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
fig2.suptitle("Cluster_after")
##plt.show()

test_X1 = np.random.normal(0, 3, 100)
test_Y1 = np.random.normal(0, 3, 100)
test_Z1 = np.random.normal(0, 3, 100)

test_X2 = np.random.normal(4, 3, 100)
test_Y2 = np.random.normal(4, 3, 100)
test_Z2 = np.random.normal(4, 3, 100)

test_X3 = np.random.normal(8, 3, 100)
test_Y3 = np.random.normal(8, 3, 100)
test_Z3 = np.random.normal(8, 3, 100)

test_X4 = np.random.normal(12, 3, 100)
test_Y4 = np.random.normal(12, 3, 100)
test_Z4 = np.random.normal(12, 3, 100)

test_X5 = np.random.normal(16, 3, 100)
test_Y5 = np.random.normal(16, 3, 100)
test_Z5 = np.random.normal(16, 3, 100)

test_X6 = np.random.normal(-5, 3, 100)
test_Y6 = np.random.normal(-5, 3, 100)
test_Z6 = np.random.normal(-5, 3, 100)

test1 = np.vstack((np.vstack((test_X1, test_Y1)), test_Z1))
test1 = np.transpose(test1)
test2 = np.vstack((np.vstack((test_X2, test_Y2)), test_Z2))
test2 = np.transpose(test2)
test3 = np.vstack((np.vstack((test_X3, test_Y3)), test_Z3))
test3 = np.transpose(test3)
test4 = np.vstack((np.vstack((test_X4, test_Y4)), test_Z4))
test4 = np.transpose(test4)
test5 = np.vstack((np.vstack((test_X5, test_Y5)), test_Z5))
test5 = np.transpose(test5)
test6 = np.vstack((np.vstack((test_X6, test_Y6)), test_Z6))
test6 = np.transpose(test6)

test1_label = np.zeros((100), dtype = np.float32)
test2_label = np.zeros((100), dtype = np.float32)
test3_label = np.zeros((100), dtype = np.float32)
test4_label = np.zeros((100), dtype = np.float32)
test5_label = np.zeros((100), dtype = np.float32)
test6_label = np.zeros((100), dtype = np.float32)

for i in range(0, test1.shape[0]):
    min_distance = 100
    for j in range(0, cluster_data.cluster_centers_.shape[0]):
        distance = math.sqrt((test1[i, 0] - cluster_data.cluster_centers_[j, 0])**2 + (test1[i, 1] - cluster_data.cluster_centers_[j, 1])**2 + (test1[i, 2] - cluster_data.cluster_centers_[j, 2])**2)
        if min_distance > distance:
            min_distance = distance
            test1_label[i] = j + 1
    if min_distance > 7:
        test1_label[i] = 0
    #print(min_distance)
print("test1")
print(test1_label)
for i in range(0, test1.shape[0]):
    min_distance = 100
    for j in range(0, cluster_data.cluster_centers_.shape[0]):
        distance = math.sqrt((test2[i, 0] - cluster_data.cluster_centers_[j, 0])**2 + (test2[i, 1] - cluster_data.cluster_centers_[j, 1])**2 + (test2[i, 2] - cluster_data.cluster_centers_[j, 2])**2)
        if min_distance > distance:
            min_distance = distance
            test2_label[i] = j + 1
    if min_distance > 7:
        test2_label[i] = 0
print("test2")
print(test2_label)
for i in range(0, test1.shape[0]):
    min_distance = 100
    for j in range(0, cluster_data.cluster_centers_.shape[0]):
        distance = math.sqrt((test3[i, 0] - cluster_data.cluster_centers_[j, 0])**2 + (test3[i, 1] - cluster_data.cluster_centers_[j, 1])**2 + (test3[i, 2] - cluster_data.cluster_centers_[j, 2])**2)
        if min_distance > distance:
            min_distance = distance
            test3_label[i] = j + 1
    if min_distance > 7:
        test3_label[i] = 0
print("test3")
print(test3_label)
for i in range(0, test1.shape[0]):
    min_distance = 100
    for j in range(0, cluster_data.cluster_centers_.shape[0]):
        distance = math.sqrt((test4[i, 0] - cluster_data.cluster_centers_[j, 0])**2 + (test4[i, 1] - cluster_data.cluster_centers_[j, 1])**2 + (test4[i, 2] - cluster_data.cluster_centers_[j, 2])**2)
        if min_distance > distance:
            min_distance = distance
            test4_label[i] = j + 1
    if min_distance > 7:
        test4_label[i] = 0
print("test4")
print(test4_label)
for i in range(0, test1.shape[0]):
    min_distance = 100
    for j in range(0, cluster_data.cluster_centers_.shape[0]):
        distance = math.sqrt((test5[i, 0] - cluster_data.cluster_centers_[j, 0])**2 + (test5[i, 1] - cluster_data.cluster_centers_[j, 1])**2 + (test5[i, 2] - cluster_data.cluster_centers_[j, 2])**2)
        if min_distance > distance:
            min_distance = distance
            test5_label[i] = j + 1
    if min_distance > 7:
        test5_label[i] = 0
print("test5")
print(test5_label)
for i in range(0, test1.shape[0]):
    min_distance = 100
    for j in range(0, cluster_data.cluster_centers_.shape[0]):
        distance = math.sqrt((test6[i, 0] - cluster_data.cluster_centers_[j, 0])**2 + (test6[i, 1] - cluster_data.cluster_centers_[j, 1])**2 + (test6[i, 2] - cluster_data.cluster_centers_[j, 2])**2)
        if min_distance > distance:
            min_distance = distance
            test6_label[i] = j + 1
    if min_distance > 7:
        test6_label[i] = 0
print("test6")
print(test6_label)

ax.legend()
ax2.legend()
plt.show()