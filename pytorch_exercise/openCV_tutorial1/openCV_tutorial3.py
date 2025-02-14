import cv2
import numpy as np
import random

img = cv2.imread("openCV_tutorial/nn_network.jpeg")
#print(img.shape)

newImg = img[:300, 100:500] #切割圖片

#img = np.empty((300, 300, 3), np.uint8) #unit代表正整數，8代表0-255要用8個bits表示
for row in range(300):
    for col in range(img.shape[1]):
        img[row][col] = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] #顏色順序是 B G R
        
cv2.imshow("img", img)
cv2.waitKey(0)