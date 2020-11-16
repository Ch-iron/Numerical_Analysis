import cv2
import os

for filePath in sorted(os.listdir("pattern")):
    fileExt = os.path.splitext(filePath)[1]
    if fileExt in [".jpg", ".jpeg"]:
        imagePath = os.path.join("pattern", filePath)
        image = cv2.imread(imagePath)
        image = cv2.resize(image, dsize=(450, 450), interpolation=cv2.INTER_LINEAR)
        cv2.imwrite(filePath, image)