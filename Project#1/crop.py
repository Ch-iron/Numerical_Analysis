    # Import necessary packages
from __future__ import print_function
import os
import sys
import cv2
import numpy as np

    # Read images from the directory
def readImages(path):
    print("Reading images from " + path, end="...")
    # List all files in the directory and read points from text files one by one
    for filePath in sorted(os.listdir(path)):
        # Add to array of images
        imagePath = os.path.join(path, filePath)
        src = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        im = cv2.resize(src, dsize=(32, 32), interpolation=cv2.INTER_AREA)
        cv2.imwrite("test/" + filePath, im)

if __name__ == '__main__':

    # Directory containing images
    dirName = "test_origin"

    # Read images
    images = readImages(dirName)