import os
import sys
import cv2
import numpy as np

# Read images from the directory
def readTestImages(path):
    print("Reading images from " + path, end="...")
    # Create array of array of images.
    i = 0
    images = np.zeros((126, 1024), dtype=np.float32)
    print(images.shape)
    # List all files in the directory and read points from text files one by one
    for filePath in sorted(os.listdir(path)):
        fileExt = os.path.splitext(filePath)[1]
        if fileExt in [".jpg", ".jpeg"]:

            # Add to array of images
            imagePath = os.path.join(path, filePath)
            im = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            im = im.flatten()
            mean = im.mean()
            im = im - mean

            if im is None :
                print("image:{} not read properly".format(imagePath))
            else :
                # Add image to list
                images[i,:] = im
                i = i + 1
    
    numImages = int(len(images))
    # Exit if no image found
    if numImages == 0 :
        print("No images found")
        sys.exit(0)

    print(str(numImages) + " files read.")
    return images

images = readTestImages("test")