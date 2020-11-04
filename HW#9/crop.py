import cv2

image = cv2.imread("image/background.jpg")
image = cv2.resize(image, dsize=(432, 304), interpolation=cv2.INTER_LINEAR)
cv2.imwrite("background_crop.jpg", image)