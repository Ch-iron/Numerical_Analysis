from __future__ import print_function
import os
import sys
import cv2
import numpy as np

def readImages(path):
    print("Reading images from " + path, end="...")
    count = 1
    # List all files in the directory and read points from text files one by one
    for filePath in sorted(os.listdir(path)):
        print(count)

        if("_0001" in filePath):
            im = cv2.imread("training/" + filePath, cv2.IMREAD_GRAYSCALE)
            cv2.imwrite("eigenfaces/" + filePath, im)
        count = count + 1

def filenameModify(path):
    print("Reading images from " + path, end="...")
    # List all files in the directory and read points from text files one by one
    for filePath in sorted(os.listdir(path)):
        fileExt = os.path.splitext(filePath)[0]
        filename = fileExt.split('.')
        im = cv2.imread("test/" + filePath, cv2.IMREAD_GRAYSCALE)
        cv2.imwrite("test_modify/" + filename[0] + "_" + filename[1] + ".jpg", im)

#readImages("training")
filenameModify("test")