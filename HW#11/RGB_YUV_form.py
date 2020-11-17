import cv2
import os
import numpy as np
import math

for filePath in sorted(os.listdir("origin")):
    fileName = os.path.splitext(filePath)[0]
    imagePath = os.path.join("origin", filePath)
    src = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    dst = src.copy()
    yuv_image = cv2.cvtColor(dst, cv2.COLOR_BGR2YUV)
    dst = np.uint64(dst)
    yuv_image = np.uint64(yuv_image)
    b, g, r = cv2.split(dst)
    y, u, v = cv2.split(yuv_image)

    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)
    b_var = np.var(b)
    g_var = np.var(g)
    r_var = np.var(r)
    b_sig = math.sqrt(b_var)
    g_sig = math.sqrt(g_var)
    r_sig = math.sqrt(r_var)

    y_mean = np.mean(y)
    u_mean = np.mean(u)
    v_mean = np.mean(v)
    y_var = np.var(y)
    u_var = np.var(u)
    v_var = np.var(v)
    y_sig = math.sqrt(y_var)
    u_sig = math.sqrt(u_var)
    v_sig = math.sqrt(v_var)

    #BG
    exp = 0
    for i in range(0, b.shape[0]):
        for j in range(0, b.shape[1]):
            exp = exp + ((b[i, j] - b_mean) * (g[i, j] - g_mean))
    cov = exp/(b.shape[0] * b.shape[1])
    corr = cov/(b_sig * g_sig)
    print(corr, end=" ")
    print("BG")

    #GR
    exp = 0
    for i in range(0, b.shape[0]):
        for j in range(0, b.shape[1]):
            exp = exp + ((g[i, j] - g_mean) * (r[i, j] - r_mean))
    cov = exp/(b.shape[0] * b.shape[1])
    corr = cov/(g_sig * r_sig)
    print(corr, end=" ")
    print("GR")

    #BR
    exp = 0
    for i in range(0, b.shape[0]):
        for j in range(0, b.shape[1]):
            exp = exp + ((b[i, j] - b_mean) * (r[i, j] - r_mean))
    cov = exp/(b.shape[0] * b.shape[1])
    corr = cov/(b_sig * r_sig)
    print(corr, end=" ")
    print("BR")

    #GY
    exp = 0
    for i in range(0, b.shape[0]):
        for j in range(0, b.shape[1]):
            exp = exp + ((g[i, j] - g_mean) * (y[i, j] - y_mean))
    cov = exp/(b.shape[0] * b.shape[1])
    corr = cov/(g_sig * y_sig)
    print(corr, end=" ")
    print("GY")

    #YU
    exp = 0
    for i in range(0, b.shape[0]):
        for j in range(0, b.shape[1]):
            exp = exp + ((y[i, j] - y_mean) * (u[i, j] - u_mean))
    cov = exp/(b.shape[0] * b.shape[1])
    corr = cov/(y_sig * u_sig)
    print(corr, end=" ")
    print("YU")

    #YV
    exp = 0
    for i in range(0, b.shape[0]):
        for j in range(0, b.shape[1]):
            exp = exp + ((y[i, j] - y_mean) * (v[i, j] - v_mean))
    cov = exp/(b.shape[0] * b.shape[1])
    corr = cov/(y_sig * v_sig)
    print(corr, end=" ")
    print("YV")

    #UV
    exp = 0
    for i in range(0, b.shape[0]):
        for j in range(0, b.shape[1]):
            exp = exp + ((u[i, j] - u_mean) * (v[i, j] - v_mean))
    cov = exp/(b.shape[0] * b.shape[1])
    corr = cov/(u_sig * v_sig)
    print(corr, end=" ")
    print("UV")

    print("----------------------------------" + fileName)