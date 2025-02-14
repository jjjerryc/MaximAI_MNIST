#常用function
import cv2
import numpy as np

img = cv2.imread("openCV_tutorial/nn_network.jpeg")
img = cv2.resize(img, (0, 0), fx = 0.8, fy = 0.8)

#把RGB圖片轉成灰階圖片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#把圖片高斯模糊
blur = cv2.GaussianBlur(img, (7, 7), 10) #第二個參數是和，必須為奇數

#取得圖片邊緣：像素值差別很大會導致看起來像一個邊緣，算法是依據和周圍像素值差別的大小幫每個像素點計算一個分數，差別大分數高
canny = cv2.Canny(img, 150, 200) #過濾差別分數低於150的，高於200分全當成邊緣看

#膨脹效果（邊緣線條變粗）
kernel = np.ones((3, 3), np.uint8) #創建一個每個值都是1，用8位元表示，大小是（3， 3）的二維陣列
kernel1 = np.ones((3, 3), np.uint8)
dilate = cv2.dilate(canny, kernel, iterations = 1)

#侵蝕效果（把圖片變細）
erode = cv2.erode(dilate, kernel1, iterations = 2)

cv2.imshow("img", img)
cv2.imshow("gray", gray)
cv2.imshow("blur", blur)
cv2.imshow("canny", canny)
cv2.imshow("dileate", dilate)
cv2.imshow("erode", erode)
cv2.waitKey(0)