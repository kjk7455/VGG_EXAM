import cv2
from Package.Finder import Finder
from datetime import datetime

cap = cv2.VideoCapture(-1)
finder = Finder()

while True:
    ret, frame = cap.read()

    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = gray[0:200, 60:260]
        if cv2.waitKey(1) & 0xFF == ord('a'):
            start_time = datetime.now()
            #name = str(start_time)
            #name += '.png'
            #finder.find(gray)
            #cv2.imwrite(name, gray)
            now = datetime.now() - start_time
            print('time : ', now)
        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
