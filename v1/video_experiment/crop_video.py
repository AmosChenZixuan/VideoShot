import cv2 as cv

if __name__ == '__main__':
    cap1 = cv.VideoCapture('/home/lyz/Documents/3.mp4')
    fps = cap1.get(5)
    size = (int(cap1.get(3)), int(cap1.get(4)))

    codec = cv.VideoWriter_fourcc('X', 'V', 'I', 'D')
    videoWriter = cv.VideoWriter('3.avi', codec, 30, size)
    start = 0
    end = cap1.get(7) - 50

    i = 0
    while 1:

        s_f, frame = cap1.read()
        if s_f:
            i += 1

            if start <= i <= end:
                print(i)
                # videoWriter.write(frame[250: -250, 500:-500])
                videoWriter.write(frame)

        else:
            break
    cap1.release()
