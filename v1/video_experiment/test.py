import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def get_video_info_and_header(path):
    cap = cv.VideoCapture(path)
    height, width, frames_num, frames_per_sec = cap.get(3), cap.get(4), cap.get(7), cap.get(5)
    return height, width, frames_num, frames_per_sec, cap


def get_average_pixel_value(cap):
    num_frames = cap.get(7)
    pix_value_average = []

    i = 0
    start_frame = 1000
    while cap.isOpened():
        ret, frame = cap.read()

        # for specific
        frame = frame[250: -250, 500:-500]
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        HSV_img = cv.cvtColor(frame,cv.COLOR_BGR2HSV)

        ret, gray = cv.threshold(gray, 127, 255, 0)

        cv.imshow('1', HSV_img[:, :, 1])
        cv.waitKey(1)
        print(HSV_img[:, :, 1].mean())
        img_grey = frame.mean(-1)
        img_grey = frame.mean(-1)
        pix_value_average.append(img_grey.mean())

        if i >= num_frames - 100:
            break
        print(i)
        i += 1
    tem_save = np.stack(pix_value_average, 0)
    np.save('pixel_values', tem_save)
    return tem_save


def clean_outsider(values):
    values = values[values < 200]
    values = values[values > 50]

    return values


def find_candidates(values):
    mean_ = np.mean(pix_value_average)
    med = np.median(pix_value_average)
    mid = np.argmax(np.bincount(pix_value_average.astype(int)))

    values_ = np.sort(values.reshape(-1, 3).mean(-1))
    max_ = values_[int(values_.shape[0] * 0.95)]
    i = 0
    flag = 0
    tmp_list = []
    rs_list = []
    print(mean_, med, mid, max_)

    while i < values.shape[0]:
        if values[i] > mean_ + (max_ - mean_) * 0.9:
            if not flag:
                flag = 1
                tmp_list.clear()
            tmp_list.append(i)
        else:
            if flag:
                if len(tmp_list) >= 30:  # 1sec
                    rs_list.append(tmp_list.copy())
                flag = 0

        print(i)
        i += 1

    return rs_list


if __name__ == '__main__':
    height, width, frames_num, frames_per_sec, cap = get_video_info_and_header('/home/lyz/Documents/3.mp4')
    height, width, frames_num, frames_per_sec = map(int, (height, width, frames_num, frames_per_sec))
    print(height, width, frames_num, frames_per_sec)

    i = 0
    # for i in range(frames_num):
    #     cap.set(cv.CAP_PROP_POS_FRAMES, i)
    #
    pix_value_average = get_average_pixel_value(cap)
