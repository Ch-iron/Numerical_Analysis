import cv2
import os
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth

## Convert Lab-1976 format
'''
for filePath in sorted(os.listdir("origin")):
    fileName = os.path.splitext(filePath)[0]
    imagePath = os.path.join("origin", filePath)
    src = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    dst = cv2.cvtColor(src, cv2.COLOR_BGR2Lab)
    L, a, b = cv2.split(dst)
    pathL = "Lab/" + fileName + "_L.jpg"
    patha = "Lab/" + fileName + "_a.jpg"
    pathb = "Lab/" + fileName + "_b.jpg"
    cv2.imwrite(pathL, L)
    cv2.imwrite(patha, a)
    cv2.imwrite(pathb, b)
'''

# Mean-shift Clustering
src = cv2.imread("origin/Image2_crop.jpg", cv2.IMREAD_COLOR)
dst = cv2.cvtColor(src, cv2.COLOR_BGR2Lab)
z = dst.reshape(-1, 3)
print(z.shape)

#bandwidth = estimate_bandwidth(z, quantile=0.2)
#print(bandwidth)

ms = MeanShift(bandwidth=9)

ms.fit(z)
cluster_centers = ms.cluster_centers_
#print("Center of clusters : ", cluster_centers)

labels = ms.labels_
print(labels)
print(labels.shape)
copy = z.copy()

for i in range(0, z.shape[0]):
    copy[i] = cluster_centers[labels[i]]
reco = copy.reshape((dst.shape))
bgr = cv2.cvtColor(reco, cv2.COLOR_Lab2BGR)
cv2.imwrite("Clustering/Image2_MS.jpg", bgr)

labels_unique = np.unique(labels)
n_clusters = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters)


## K-means Clustering
'''
src = cv2.imread("origin/Image1.jpg", cv2.IMREAD_COLOR)
dst = cv2.cvtColor(src, cv2.COLOR_BGR2Lab)
z = dst.reshape(-1, 3)
print(z.shape)
'''
z = np.float32(z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS

k = n_clusters

coompact, labels, centers = cv2.kmeans(z, k, None, criteria, 10, flags)

centers = np.uint8(centers)
res = centers[labels.flatten()]
res2 = res.reshape((dst.shape))
#print(res2.shape)

bgr = cv2.cvtColor(res2, cv2.COLOR_Lab2BGR)

cv2.imwrite("Clustering/Image2_K.jpg", bgr)
