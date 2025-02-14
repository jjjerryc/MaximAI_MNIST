import cv2

#cap = cv2.VideoCapture("/Users/jayson/Documents/VSCODE/python/openCV_tutorial/jarrod.mp4") #讀影片
cap = cv2.VideoCapture(0) #讀鏡頭的話輸入括號直接輸入0，外接鏡頭輸入1

while True:
    ret, frame = cap.read() #有沒有讀到下一張的boolean（變數ret），成功的話回傳下一張圖片（變數frame）
    frsme = cv2.resize(frame, (0, 0), fx = 0.8, fy = 0.8) #調整尺寸
    if ret:
        cv2.imshow("video", frame)
    else:
        break 
    if cv2.waitKey(10) == ord("q"): #按q跳出去
        break