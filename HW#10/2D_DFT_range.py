import numpy as np
import cv2
import random
import os
import re

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(precision=4)

#select 210 dominant coefficient from dft of each pattern
#make average DFT coefficient
dft_collect = np.zeros((5, 10, 21), dtype=np.float32)
average_dft = np.zeros((20, 10, 21), dtype=np.float32)
difference = np.zeros(5, dtype=np.float32)
minmax = np.zeros((20), dtype=np.float32)
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
        dft_collect[k, 9, 10] = 0
        for i in range(0, 10):
            for j in range(0, 21):
                if dft_collect[k, i, j] < 150:
                    dft_collect[k, i, j] = 0
    mean = mean/5
    mean[9, 10] = 0
    for i in range(0, 10):
        for j in range(0, 21):
            if mean[i, j] > 150:
                average_dft[n, i, j] = part[i, j]
    for i in range(0, 5):
        difference[i] = np.sum(abs(dft_collect[i] - average_dft[n]))
    minmax[n] = np.max(difference)
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
        similar = np.zeros(20, dtype=int)
        count = 0
        for i in range(0, 20):
            compare = part - average_dft[i]
            compare_sum[i] = np.sum(abs(compare))
            if compare_sum[i] <= minmax[i]:
                print("This is " + str(i + 1) + " Pattern!! " + str(k))
                count = count + 1
                similar[count - 1] = i
        if count == 0:
            print("Not exist matching Pattern" + str(k))
        elif count > 1:
            min_distance = compare_sum[similar[0]]
            min_arg = similar[0]
            for i in range(1, count):
                if compare_sum[similar[i]] < min_distance:
                    min_distance = compare_sum[similar[i]]
                    min_arg = similar[i]
            print("Finally, This is " + str(min_arg + 1) +  "Pattern!!!!!!!!!!!!!!!!!")