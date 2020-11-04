import cv2

b = cv2.imread("back_bgr/origin_b.bmp", cv2.COLOR_BGR2GRAY)
g = cv2.imread("back_bgr/origin_g.bmp", cv2.COLOR_BGR2GRAY)
r = cv2.imread("back_bgr/origin_r.bmp", cv2.COLOR_BGR2GRAY)

inversebgr = cv2.merge((b, g, r))
cv2.imwrite("recover_background.bmp", inversebgr)