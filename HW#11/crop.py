import cv2
import os

image = cv2.imread("origin/image06.jpg")
image = cv2.resize(image, dsize=(600, 400), interpolation=cv2.INTER_LINEAR)]
cv2.imwrite("origin/image06.jpg", image)
