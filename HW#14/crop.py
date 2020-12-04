import cv2
import os

image = cv2.imread("origin/Image3.jpg")
image = cv2.resize(image, dsize=(200, 120), interpolation=cv2.INTER_LINEAR)
cv2.imwrite("origin/Image3_crop.jpg", image)
