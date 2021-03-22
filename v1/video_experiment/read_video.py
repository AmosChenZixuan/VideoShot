import cv2 as cv


cap1 = cv.VideoCapture('video1.avi')

while 1 :
    s, f = cap1.read()
    if not s:
        break
    else:
        cv.imshow('1', f)
        cv.waitKey(1)
