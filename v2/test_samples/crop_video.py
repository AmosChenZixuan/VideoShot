import cv2 as cv

ori = './3.avi'
target = './c3.avi'

cap = cv.VideoCapture(ori)
x, y, w, h = 450, 280, 900, 550
size = w, h
frame_per_sec = cap.get(5)
print(frame_per_sec)

fourcc = cv.VideoWriter_fourcc(*'XVID')

writer = cv.VideoWriter(target, fourcc, frame_per_sec, size)
# out = cv.VideoWriter('testwrite.avi', fourcc, 20.0, (1920, 1080), True)
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        frame = frame[y:y+h, x:x+w]
        # print(size, frame.shape)
        writer.write(frame)
        cv.imshow("capture", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print('no returns')
        break
cap.release()
writer.release()
cv.destroyAllWindows()