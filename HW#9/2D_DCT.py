import cv2
import numpy as np
import math

np.set_printoptions(threshold=np.inf)
np.set_printoptions(linewidth=np.inf)
#np.set_printoptions(precision=4)

#BGR split
n = 16
src = cv2.imread("background_crop.jpg", cv2.IMREAD_COLOR)
dst = src.copy()
b, g, r = cv2.split(dst)

#B
#16*16 divide
partition = np.zeros((b.shape[0]//n * b.shape[1]//n, n, n), dtype=np.float32)
i = 0
for av in range(0, b.shape[0]//n):
    for ah in range(0, b.shape[1]//n):
        partition[i] = b[n * av : n * (av + 1), n * ah : n * (ah + 1)]
        i = i + 1

#2D DCT
f = np.zeros((b.shape[0]//n * b.shape[1]//n, n, n), dtype=np.float32)
abf = np.zeros((b.shape[0]//n * b.shape[1]//n, n * n), dtype=np.float32)
count = 1
for i in range(0, f.shape[0]):
    for u in range(0, n):
        for v in range(0, n):
            for x in range(0, n):
                for y in range(0, n):
                    if u == 0:
                        cu = math.sqrt(1/2)
                    else:
                        cu = 1
                    if v == 0:
                        cv = math.sqrt(1/2)
                    else:
                        cv = 1
                    fuv= 1/8 * cu * cv * partition[i, x, y] * math.cos(v * math.pi * (2 * y + 1)/(2 * n)) * math.cos(u * math.pi * (2 * x + 1)/(2 * n))
                    f[i, u, v] = f[i, u, v] + fuv
    abf[i] = abs(f[i]).flatten()
    abf[i] = np.sort(abf[i])[::-1]
    if i % 100 == 0:
        print("16*16 finish", end=" ")
        print(i)
#choice highest 16 coeffcient
for i in range(0, f.shape[0]):
    for j in range(0, n):
        for k in range(0, n):
            if abs(f[i, j, k]) < abf[i, 15]:
                f[i, j, k] = 0
#print(f[100])
#Inverse DCT
s = np.zeros((b.shape[0]//n * b.shape[1]//n, n, n), dtype=np.float32)
for i in range(0, s.shape[0]):
    for x in range(0, n):
        for y in range(0, n):
            for u in range(0, n):
                for v in range(0, n):
                    if u == 0:
                        cu = math.sqrt(1/2)
                    else:
                        cu = 1
                    if v == 0:
                        cv = math.sqrt(1/2)
                    else:
                        cv = 1
                    sxy= 1/8 * cu * cv * f[i, u, v] * math.cos(v * math.pi * (2 * y + 1)/(2 * n)) * math.cos(u * math.pi * (2 * x + 1)/(2 * n))
                    s[i, x, y] = s[i, x, y] + sxy
            if s[i, x, y] < 0:
                s[i, x, y] = 0
            elif s[i, x, y] > 255:
                s[i, x, y] = 255
#print(s[100])
# recover origin image
hadd = np.zeros((b.shape[0]//n, n, b.shape[1]), dtype=np.float32)
for i in range(0, b.shape[0]//n):
    add = s[i * b.shape[1]//n]
    for j in range(0, b.shape[1]//n - 1):
        add = cv2.hconcat([add, s[i * b.shape[1]//n + j + 1]])
    hadd[i] = add
origin = hadd[0]
for i in range(0, b.shape[0]//n - 1):
    origin = cv2.vconcat([origin, hadd[i + 1]])
cv2.imwrite("back_bgr/origin_b.bmp", origin)

#G
partition = np.zeros((b.shape[0]//n * b.shape[1]//n, n, n), dtype=np.float32)
i = 0
for av in range(0, b.shape[0]//n):
    for ah in range(0, b.shape[1]//n):
        partition[i] = g[n * av : n * (av + 1), n * ah : n * (ah + 1)]
        i = i + 1

f = np.zeros((b.shape[0]//n * b.shape[1]//n, n, n), dtype=np.float32)
abf = np.zeros((b.shape[0]//n * b.shape[1]//n, n * n), dtype=np.float32)
count = 1
for i in range(0, f.shape[0]):
    for u in range(0, n):
        for v in range(0, n):
            for x in range(0, n):
                for y in range(0, n):
                    if u == 0:
                        cu = math.sqrt(1/2)
                    else:
                        cu = 1
                    if v == 0:
                        cv = math.sqrt(1/2)
                    else:
                        cv = 1
                    fuv= 1/8 * cu * cv * partition[i, x, y] * math.cos(v * math.pi * (2 * y + 1)/(2 * n)) * math.cos(u * math.pi * (2 * x + 1)/(2 * n))
                    f[i, u, v] = f[i, u, v] + fuv
    abf[i] = abs(f[i]).flatten()
    abf[i] = np.sort(abf[i])[::-1]
    if i % 100 == 0:
        print("16*16 finish", end=" ")
        print(i)
for i in range(0, f.shape[0]):
    for j in range(0, n):
        for k in range(0, n):
            if abs(f[i, j, k]) < abf[i, 15]:
                f[i, j, k] = 0
#print(f[100])
s = np.zeros((b.shape[0]//n * b.shape[1]//n, n, n), dtype=np.float32)
for i in range(0, s.shape[0]):
    for x in range(0, n):
        for y in range(0, n):
            for u in range(0, n):
                for v in range(0, n):
                    if u == 0:
                        cu = math.sqrt(1/2)
                    else:
                        cu = 1
                    if v == 0:
                        cv = math.sqrt(1/2)
                    else:
                        cv = 1
                    sxy= 1/8 * cu * cv * f[i, u, v] * math.cos(v * math.pi * (2 * y + 1)/(2 * n)) * math.cos(u * math.pi * (2 * x + 1)/(2 * n))
                    s[i, x, y] = s[i, x, y] + sxy
            if s[i, x, y] < 0:
                s[i, x, y] = 0
            elif s[i, x, y] > 255:
                s[i, x, y] = 255
#print(s[100])

hadd = np.zeros((b.shape[0]//n, n, b.shape[1]), dtype=np.float32)
for i in range(0, b.shape[0]//n):
    add = s[i * b.shape[1]//n]
    for j in range(0, b.shape[1]//n - 1):
        add = cv2.hconcat([add, s[i * b.shape[1]//n + j + 1]])
    hadd[i] = add
origin = hadd[0]
for i in range(0, b.shape[0]//n - 1):
    origin = cv2.vconcat([origin, hadd[i + 1]])
cv2.imwrite("back_bgr/origin_g.bmp", origin)

#R
partition = np.zeros((b.shape[0]//n * b.shape[1]//n, n, n), dtype=np.float32)
i = 0
for av in range(0, b.shape[0]//n):
    for ah in range(0, b.shape[1]//n):
        partition[i] = r[n * av : n * (av + 1), n * ah : n * (ah + 1)]
        i = i + 1

f = np.zeros((b.shape[0]//n * b.shape[1]//n, n, n), dtype=np.float32)
abf = np.zeros((b.shape[0]//n * b.shape[1]//n, n * n), dtype=np.float32)
count = 1
for i in range(0, f.shape[0]):
    for u in range(0, n):
        for v in range(0, n):
            for x in range(0, n):
                for y in range(0, n):
                    if u == 0:
                        cu = math.sqrt(1/2)
                    else:
                        cu = 1
                    if v == 0:
                        cv = math.sqrt(1/2)
                    else:
                        cv = 1
                    fuv= 1/8 * cu * cv * partition[i, x, y] * math.cos(v * math.pi * (2 * y + 1)/(2 * n)) * math.cos(u * math.pi * (2 * x + 1)/(2 * n))
                    f[i, u, v] = f[i, u, v] + fuv
    abf[i] = abs(f[i]).flatten()
    abf[i] = np.sort(abf[i])[::-1]
    if i % 100 == 0:
        print("16*16 finish", end=" ")
        print(i)
for i in range(0, f.shape[0]):
    for j in range(0, n):
        for k in range(0, n):
            if abs(f[i, j, k]) < abf[i, 15]:
                f[i, j, k] = 0
#print(f[100])
s = np.zeros((b.shape[0]//n * b.shape[1]//n, n, n), dtype=np.float32)
for i in range(0, s.shape[0]):
    for x in range(0, n):
        for y in range(0, n):
            for u in range(0, n):
                for v in range(0, n):
                    if u == 0:
                        cu = math.sqrt(1/2)
                    else:
                        cu = 1
                    if v == 0:
                        cv = math.sqrt(1/2)
                    else:
                        cv = 1
                    sxy= 1/8 * cu * cv * f[i, u, v] * math.cos(v * math.pi * (2 * y + 1)/(2 * n)) * math.cos(u * math.pi * (2 * x + 1)/(2 * n))
                    s[i, x, y] = s[i, x, y] + sxy
            if s[i, x, y] < 0:
                s[i, x, y] = 0
            elif s[i, x, y] > 255:
                s[i, x, y] = 255
#print(s[100])

hadd = np.zeros((b.shape[0]//n, n, b.shape[1]), dtype=np.float32)
for i in range(0, b.shape[0]//n):
    add = s[i * b.shape[1]//n]
    for j in range(0, b.shape[1]//n - 1):
        add = cv2.hconcat([add, s[i * b.shape[1]//n + j + 1]])
    hadd[i] = add
origin = hadd[0]
for i in range(0, b.shape[0]//n - 1):
    origin = cv2.vconcat([origin, hadd[i + 1]])
cv2.imwrite("back_bgr/origin_r.bmp", origin)