import numpy as np
import cv2
import random
import os
import re

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=4)

#random select 5 block each pattern
'''
for filePath in sorted(os.listdir("pattern")):
    fileExt = os.path.splitext(filePath)[0]
    for i in range(0, 5):
        imagePath = os.path.join("pattern", filePath)
        image = cv2.imread(imagePath)
        rand = random.randrange(0, image.shape[0] - 64)
        image = image[rand: rand + 64, rand : rand + 64]
        filename = fileExt + "-" + str(i + 1)
        filename = filename + ".jpg"
        storePath = os.path.join("pattern_random", filename)
        cv2.imwrite(storePath, image)
'''

#select 210 dominant coefficient from dft of each pattern
'''
dft_collect = np.zeros((20, 10, 21), dtype=np.float32)
for filePath in os.listdir("pattern_crop"):
    filename = os.path.splitext(filePath)[0]
    numbers = int(re.findall("\d+", filename)[0])
    path = os.path.join("pattern_crop", filePath)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    dft = np.fft.fft2(img)
    dft = np.fft.fftshift(dft)
    magnitude = 20 * np.log(np.abs(dft))

    part = magnitude[23:33, 22:43]
    part[9, 10] = 0
    dft_collect[numbers - 1] = part
    #for i in range(0, 10):
    #    for j in range(0, 21):
    #        if part[i, j] > 150:
    #            dft_collect[numbers - 1, i, j] = part[i, j]
'''
#select 210 dominant coefficient from dft of each pattern
#make average DFT coefficient
dft_collect = np.zeros((5, 10, 21), dtype=np.float32)
average_dft = np.zeros((20, 10, 21), dtype=np.float32)
n = 0
for filePath in os.listdir("pattern"):
    mean = np.zeros((10, 21), dtype=np.float32)
    for k in range(0, 5):
        imagePath = os.path.join("pattern", filePath)
        image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        rand = random.randrange(0, image.shape[0] - 64)
        img = image[rand: rand + 64, rand : rand + 64]
    
        dft = np.fft.fft2(img)
        dft = np.fft.fftshift(dft)
        magnitude = 20 * np.log(np.abs(dft))

        part = magnitude[23:33, 22:43]
        dft_collect[k] = part
        mean = mean + dft_collect[k]
    mean = mean/5
    mean[9, 10] = 0
    for i in range(0, 10):
        for j in range(0, 21):
            if mean[i, j] > 150:
                average_dft[n, i, j] = mean[i, j]
    n = n + 1

#random select 5 block each pattern
#compare random block with dft of each pattern by distance
for filePath in sorted(os.listdir("pattern")):
    for k in range(0, 5):
        imagePath = os.path.join("pattern", filePath)
        image = cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)
        rand = random.randrange(0, image.shape[0] - 64)
        img = image[rand: rand + 64, rand : rand + 64]

#for filePath in os.listdir("pattern_random"):
#    path = os.path.join("pattern_random", filePath)
#    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        dft = np.fft.fft2(img)
        dft = np.fft.fftshift(dft)
        magnitude = 20 * np.log(np.abs(dft))
        part = magnitude[23:33, 22:43]
        part[9, 10] = 0
        for i in range(0, 10):
            for j in range(0, 21):
                if part[i, j] < 150:
                    part[i, j] = 0

        compare_sum = np.zeros(20, dtype=np.float32)
        for i in range(0, 20):
            compare = part - average_dft[i]
            compare_sum[i] = np.sum(abs(compare))
        match = np.argmin(compare_sum)
        print("This is " + str(match + 1) + " Pattern!!")