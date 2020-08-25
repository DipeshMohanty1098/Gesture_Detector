import cv2

cap = cv2.VideoCapture('D:/train/videos/ok.mp4')

i = 1

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('D:/train/ok/ok.{}.jpg'.format(i),frame)
    i+=1

cap.release()
cv2.destroyAllWindows()
