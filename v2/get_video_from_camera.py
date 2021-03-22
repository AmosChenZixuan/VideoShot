import cv2 as cv

cap = cv.VideoCapture(0)
size = (int(cap.get(3)), int(cap.get(4)))
frame_per_sec = cap.get(5)

fourcc = cv.VideoWriter_fourcc(*'XVID')

writer = cv.VideoWriter('./tmp_video/testwrite3.avi', fourcc, frame_per_sec, size, True)
# out = cv.VideoWriter('testwrite.avi', fourcc, 20.0, (1920, 1080), True)
while (1):
    ret, frame = cap.read()
    writer.write(frame)
    cv.imshow("capture", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
writer.release()
cv.destroyAllWindows()
