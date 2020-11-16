import cv2
import os

'''
for filePath in sorted(os.listdir("pattern")):
    fileExt = os.path.splitext(filePath)[1]
    if fileExt in [".jpg", ".jpeg"]:
        imagePath = os.path.join("pattern", filePath)
        image = cv2.imread(imagePath)
        image = image[225: 225 + 64, 225 : 225 + 64]
        cv2.imwrite(filePath, image)
'''
image = cv2.imread("pattern/pattern6.jpg")
#image = cv2.resize(image, dsize=(390, 390), interpolation=cv2.INTER_LINEAR)
image = image[200 : 264, 200 : 264]
cv2.imwrite("pattern_crop/pattern6.jpg", image)
