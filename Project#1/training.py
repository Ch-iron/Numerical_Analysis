# Import necessary packages
from __future__ import print_function
import os
import sys
import cv2
import numpy as np

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=4)

# Create data matrix from a list of images
def createMatrixA(images):
    print("Creating data matrix",end=" ... ")

    numImages = len(images)
    sz = images[0].shape
    matrix_a = np.zeros((numImages, sz[0] * sz[1]), dtype=np.float32)
    for i in range(0, numImages):
        image = images[i].flatten()
        matrix_a[i,:] = image

    print("DONE")
    return matrix_a

# Read images from the directory
def readImages(path):
    print("Reading images from " + path, end="...")
    # Create array of array of images.
    images = []
    # List all files in the directory and read points from text files one by one
    for filePath in sorted(os.listdir(path)):
        fileExt = os.path.splitext(filePath)[1]
        if fileExt in [".jpg", ".jpeg"]:
            # Add to array of images
            imagePath = os.path.join(path, filePath)
            im = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)

            if im is None :
                print("image:{} not read properly".format(imagePath))
            else :
                images.append(im)
    numImages = int(len(images))
    # Exit if no image found
    if numImages == 0 :
        print("No images found")
        sys.exit(0)

    print(str(numImages) + " files read.")
    return images

# Read images from the directory
def readTestImages(path):
    print("Reading images from " + path, end="...")
    # Create array of array of images.
    i = 0
    images = np.zeros((126, 32 * 32), dtype=np.float32)
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

def ImageMean(path):
    means = np.zeros((126, 32 * 32), dtype=np.float32)
    i = 0
    # List all files in the directory and read points from text files one by one
    for filePath in sorted(os.listdir(path)):
        fileExt = os.path.splitext(filePath)[1]
        if fileExt in [".jpg", ".jpeg"]:
            # Add to array of images
            imagePath = os.path.join(path, filePath)
            im = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
            mean = im.mean()
            means[i,:] = mean
            i = i + 1
    return means

if __name__ == '__main__':

    # Number of EigenFaces
    NUM_EIGEN_FACES = 16

    # Directory containing images
    dirName = "training"

    # Read images
    images = readImages(dirName)

    # Size of images
    sz = [32, 32]

    # Create data matrix for PCA.
    a_matrix = createMatrixA(images)

    # SVD for eigenvalue
    print("SVD ing", end="...")
    u, s, vt = np.linalg.svd(a_matrix)
    s = s.flatten()
    j = 1
    for i in s:
        if(i > 10000):
            print("Singular Value : " + str(i))
            print("Number of Singular Value : " + str(j))
            j = j + 1

    # Compute the eigenvectors from the stack of images created
    print("Calculating PCA ", end="...")
    mean, eigenVectors = cv2.PCACompute(a_matrix, mean=None, maxComponents=NUM_EIGEN_FACES)

    print ("DONE")

    eigenFaces = np.zeros((NUM_EIGEN_FACES, sz[0] * sz[1]), dtype=np.float32)

    i = 0
    for eigenVector in eigenVectors:
        eigenFaces[i,:] = eigenVector
        i = i + 1
        
    #Create eigenFace image    
    for i in range(0, NUM_EIGEN_FACES):
        eigenFace = eigenFaces[i].reshape(sz)
        eigenFace = eigenFace * 1000 + 128
        cv2.imwrite("eigen_test/eigenface" + str(i) + ".jpg", eigenFace)

    test_images = readTestImages("test")

    #Solve Ck on each image
    ck = np.zeros((126, NUM_EIGEN_FACES), dtype=np.float32)
    for i in range(0, 126):
        for j in range(0, NUM_EIGEN_FACES):
            ck[i, j] = np.dot(test_images[i], eigenFaces[j])
    ck3d = ck.reshape(14, 9, -1)
    count = 1
    for i in range(0, 14):
        for j in range(0, 9):
            print(ck3d[i, j])
            print(ck3d[i].mean(axis = 0))
        print("----------------------", end = ' ')
        print(count)
        count = count + 1
    
    #Two images input, decide same people
    ck_except_m = np.zeros((14, 7, 16),dtype=np.float32)
    ck_mean = np.zeros((14, NUM_EIGEN_FACES),dtype=np.float32)
    ck_mean_difference = np.zeros((14 * 13, 16), dtype=np.float32)

    for i in range(0, 14):
        for j in range(0, 16):
            tmp = np.sort(ck3d[i,:,j])
            tmp = tmp[1:8]
            ck_except_m[i,:,j] = tmp

    for i in range(0, 14):
        ck_mean[i] = ck_except_m[i].mean(axis = 0)
    print(ck_mean)

    num = 0
    for i in range(0, 14):
        for j in range(0, 14):
            if i != j:
                ck_mean_difference[num] = abs(ck_mean[i] - ck_mean[j])
                num = num + 1
    ck_mean_difference_mean = ck_mean_difference.mean(axis = 0)
    print(ck_mean_difference_mean)

    ck_compare = np.zeros((2, NUM_EIGEN_FACES), dtype=np.float32)
    compare_images = readTestImages("compare_two_images")
    for i in range(0, 2):
        for j in range(0, NUM_EIGEN_FACES):
            ck_compare[i, j] = np.dot(compare_images[i], eigenFaces[j])
    ck_difference = abs(ck_compare[0] - ck_compare[1])
    print(ck_difference)

    different_count = 0
    for i in range(0, 16):
        if ck_mean_difference_mean[i] <= ck_difference[i]:
            different_count = different_count + 1
    if different_count >= 7:
        print("Different Count : " + str(different_count))
        print("Different Person")
    else:
        print("Different Count : " + str(different_count))
        print("Same Person")
    
    #new images input, face classify
    ck_min_max = np.zeros((14, 2, NUM_EIGEN_FACES),dtype=np.float32)
    ck_column = np.zeros((9,), dtype=np.float32)
    ck_final = np.zeros((7,), dtype=np.float32)
    for i in range(0, 14):
        for j in range(0, 16):
            ck_column = ck3d[i,:,j]
            ck_column = np.sort(ck_column)
            ck_final = ck_column[1:7]
            ck_min_max[i,0,j] = np.max(ck_final)
            ck_min_max[i,1,j] = np.min(ck_final)
    for i in range(0, 14):
        print(ck_min_max[i])

    ck_new = np.zeros((14, NUM_EIGEN_FACES), dtype=np.float32)
    new_images = readTestImages("new")
    for i in range(0, 14):
        for j in range(0, NUM_EIGEN_FACES):
            ck_new[i, j] = np.dot(new_images[i], eigenFaces[j])
    
    count = 1
    for i in range(0, 14):
        true_count = 0
        for j in range(0, NUM_EIGEN_FACES):
            if ck_new[i, j] <= ck_min_max[i,0,j] and ck_new[i, j] >= ck_min_max[i,1,j]:
                if(j != 12):
                    true_count = true_count + 1
        if true_count >= 10:
            print("Matching" + str(i + 1))
        print("-------------------------------")

    #Image recovery
    means = ImageMean("test")
    for i in range(0, 126):
        recovery = np.zeros((32 * 32,), dtype=np.float32)
        for j in range(0, NUM_EIGEN_FACES):
            multiple = ck[i][j] * eigenFaces[j]
            recovery = recovery + multiple
        recovery = recovery + means[i]
        output = recovery.reshape(sz)
        cv2.imwrite("recovery/recovery_" + str(i + 1) + ".jpg", output)