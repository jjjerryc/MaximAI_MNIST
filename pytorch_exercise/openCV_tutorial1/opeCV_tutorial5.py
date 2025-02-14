import cv2
import numpy as np

img = np.zeros((600, 600, 3), np.uint8)

#畫直線
cv2.line(img, (0, 0), (img.shape[1], img.shape[0]), (255, 255 ,255), 2)
#畫方形
cv2.rectangle(img, (0, 0), (400, 300), (240, 240, 240), cv2.FILLED)
#畫圓形
cv2.circle(img, (300, 400), 30, (0, 255, 0), cv2.FILLED)
#寫文字
cv2.putText(img, 'hello', (100, 500), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255))

cv2.imshow('img', img)
cv2.waitKey(0)