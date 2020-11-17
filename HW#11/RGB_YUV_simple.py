import cv2
import os
import numpy as np

for filePath in sorted(os.listdir("origin")):
    fileName = os.path.splitext(filePath)[0]
    imagePath = os.path.join("origin", filePath)
    src = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    dst = src.copy()
    b, g, r = cv2.split(dst)
    #splitPath = os.path.join("rgb", fileName)
    #cv2.imwrite(splitPath + "_b.jpg", b)
    #cv2.imwrite(splitPath + "_g.jpg", g)
    #cv2.imwrite(splitPath + "_r.jpg", r)

    #splitPath = os.path.join("yuv", fileName)
    #cv2.imwrite(splitPath + "_y.jpg", y)
    #cv2.imwrite(splitPath + "_u.jpg", u)
    #cv2.imwrite(splitPath + "_v.jpg", v)
    
    b = b.flatten()
    g = g.flatten()
    r = r.flatten()

    yuv_image = cv2.cvtColor(dst, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv_image)
    y = y.flatten()
    u = u.flatten()
    v = v.flatten()

    bg_cor = np.corrcoef(b, g)
    gr_cor = np.corrcoef(g, r)
    br_cor = np.corrcoef(b, r)
    yg_cor = np.corrcoef(y, g)
    yu_cor = np.corrcoef(y, u)
    yv_cor = np.corrcoef(y, v)
    uv_cor = np.corrcoef(u, v)

    print(bg_cor[0, 1], end=" ")
    print("BG")
    print(gr_cor[0, 1], end=" ")
    print("GR")
    print(br_cor[0, 1], end=" ")
    print("BR")
    print(yg_cor[0, 1], end=" ")
    print("YG")
    print(yu_cor[0, 1], end=" ")
    print("YU")
    print(yv_cor[0, 1], end=" ")
    print("YV")
    print(uv_cor[0, 1], end=" ")
    print("UV")
    print("--------------------------------" + fileName)