import cv2

img = cv2.imread("/Users/jayson/Documents/VSCODE/python/openCV_tutorial/nn_network.jpeg") #讀照片

img = cv2.resize(img, (300, 300)) #調整圖片大小
img = cv2.resize(img, (0, 0), fx = 0.5, fy = 0.5)
cv2.imshow("img", img)
cv2.waitKey(0) #括號中寫入圖片持續顯示的時間（毫秒）， 0代表無限久，等待鍵盤上的任意鍵被按，回傳按鍵值結束
