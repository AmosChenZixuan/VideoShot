from retinaface.dnn_detector import RetinafaceDetector_dnn
import cv2 as cv
import pandas as pd
from detect_id_card import id_card_detect
from helpers import *
import numpy as np
from face_detectors import retina_face_detect, retina_face_distinguish


def pick_up_candi(candi, k, values):
    candi_values = values[candi]
    idx = topk_idx(candi_values, k)
    return np.array(candi)[idx]


def put_text(img, text: str):
    font = cv.FONT_HERSHEY_SIMPLEX
    return cv.putText(img, text, (100, 100), font, 1.2, (255, 255, 255), 2)


def topk_idx(v, k):
    pd_v = pd.Series(v)
    return pd_v.sort_values().index[:k].values.astype(np.int)


def get_video_info_and_header(path):
    cap = cv.VideoCapture(path)
    height, width, frames_num, frames_per_sec = cap.get(3), cap.get(4), cap.get(7), cap.get(5)
    return height, width, frames_num, frames_per_sec, cap


def get_average_pixel_value(cap):
    num_frames = cap.get(7)
    pix_value_average = []

    # start_frame = int(20 * cap.get(5))
    # start_frame = 7000
    i = START_FRAME
    cap.set(cv.CAP_PROP_POS_FRAMES, i)
    while cap.isOpened():

        ret, frame = cap.read()

        # for specific
        frame = frame[250: -250, 500:-500]

        frame_roi = frame[int(frame.shape[0] * 0.1): int(frame.shape[0] * 0.9),
                    int(frame.shape[1] * 0.1): int(frame.shape[1] * 0.9)]

        hsv_img = cv.cvtColor(frame_roi, cv.COLOR_BGR2HSV)
        pix_value_average.append(hsv_img[:, :, 1].mean())

        imgzi = put_text(frame_roi, str(pix_value_average[-1]))
        # cv.imshow('2', hsv_img)
        # cv.waitKey(1)
        if i >= num_frames - 30:
            break
        print(i)
        i += 1
    tem_save = np.stack(pix_value_average, 0)
    cv.destroyAllWindows()

    return tem_save


def clean_outsider(values):
    values = values[values < 200]
    values = values[values > 50]

    return values


def cut(values, min_, mean_, threshold_low, threshold_high, frame_length):
    tmp_list = []
    rs_list = []
    i = 0
    flag = 0
    while i < values.shape[0]:
        if min_ + (mean_ - min_) * threshold_low < values[i] < min_ + (mean_ - min_) * threshold_high:
            if not flag:
                flag = 1
                tmp_list.clear()
            tmp_list.append(i)
        else:
            if flag:
                if len(tmp_list) >= frame_length:  # 1sec
                    rs_list.append(tmp_list.copy())
                flag = 0

        print(i)
        i += 1

    ii = 0

    final_list = []
    flag = 0
    while ii < len(rs_list) - 1:
        if ii == len(rs_list) - 2:
            flag = 1
        cur_list = rs_list[ii]
        next_list = rs_list[ii + 1]
        if next_list[0] - cur_list[-1] < 20:
            final_list.append(list(range(cur_list[0], next_list[-1])))
            if flag:
                break
            ii += 2

        else:
            final_list.append(cur_list)
            if flag:
                final_list.append(rs_list[-1])
            ii += 1

    return final_list


def find_candidates(values):
    mean_ = np.mean(pix_value_average)
    med = np.median(pix_value_average)
    min_ = np.min(pix_value_average)
    mid = np.argmax(np.bincount(pix_value_average.astype(int)))

    i = 0
    flag = 0
    tmp_list = []
    rs_list = []
    print(mean_, med, mid, min_)

    paper_res_list = cut(values, min_, med, 0, 0.65, 15)
    id_res_list = cut(values, min_, med, 0.25, 0.825, 15)

    return paper_res_list, id_res_list


def search_stable_img(best_candi, values):
    top_k = topk_idx(values[best_candi], 15).tolist()
    top_k_idx = np.array(best_candi)[top_k]
    dif = [(values[i - 2] + values[i + 2] - values[i] * 2) for i in top_k_idx]
    best_i = np.argmin(np.array(dif))
    opt_idx = top_k_idx[best_i]

    return opt_idx


def find_paper(paper_candis, start_frame, values):
    weight_list = []

    for idx, candi in enumerate(paper_candis):
        if candi[0] < start_frame or start_frame in candi:
            continue
        else:
            weight_list.append((idx, values[candi].mean() * 0.5 + 0.5 * len(candi)))
    best_idx = np.argmax(np.array([x[-1] for x in weight_list]))
    best_candi = paper_candis[weight_list[best_idx][0]]
    paper_idx = search_stable_img(best_candi, values)
    return paper_idx


def find_id_card(candis, cap, values):
    id_card_index_candidate = []
    face_scores = []
    rect_args = load_model()
    for candi in candis:

        idxs = pick_up_candi(candi, 10, values)

        for idx in idxs:
            cap.set(cv.CAP_PROP_POS_FRAMES, START_FRAME + idx)
            res, frame = cap.read()
            frame = frame[250: -250, 500:-500]
            print(START_FRAME + idx)
            croped_img, rect_score = id_card_detect(frame, rect_args)

            if not croped_img.width:
                continue
            else:
                croped_img = cv.cvtColor(np.asarray(croped_img), cv.COLOR_RGB2BGR)
                half = croped_img[int(croped_img.shape[0] * 0.1): int(croped_img.shape[0] * 0.7),
                       int(croped_img.shape[1] * 0.55): int(croped_img.shape[1] * 0.95)]
                has_face, score, _ = retina_face_distinguish(half)
                if has_face:
                    id_card_index_candidate.append(idx)
                    face_scores.append(score)

    best_face_score_idx = np.argmax(np.array(face_scores))
    best_idx = id_card_index_candidate[best_face_score_idx]

    return best_idx, id_card_index_candidate


def show_candidates(candis, cap):
    for candi in candis:
        # k = 20 if 20 < len(candi) else len(candi)
        # idxs = topk_idx(candi, k) + candi[0]
        idxs = candi
        cap.set(cv.CAP_PROP_POS_FRAMES, START_FRAME + candi[0])

        for idx in idxs:
            res, frame = cap.read()
            print(START_FRAME + idx)
            cv.imshow('imgae', frame)
            if cv.waitKey(1) == ord('q'):
                break
        cv.waitKey(0)


def find_best_face(frames_num, cap):
    interval = 15
    face_idx_candis = []
    face_scores = []
    face_scores_ori = []
    front_face_score = []

    lm_test = []

    for cur_frame_idx in range(0, frames_num, interval):
        cap.set(cv.CAP_PROP_POS_FRAMES, cur_frame_idx + START_FRAME)
        print(cur_frame_idx + START_FRAME)
        s_f, frame = cap.read()
        if s_f:
            has_face, score, lm = retina_face_distinguish(frame[250: -250, 500:-500])
            if has_face and score.shape[0] == 2:
                face_idx_candis.append(cur_frame_idx)
                face_scores.append(np.min(score))
                face_scores_ori.append(score)
                front_face_score.append((lm[:, 1] - lm[:, 0]).mean())
                lm_test.append(lm)

    front_face_score_np = np.array(front_face_score)
    front_face_score_np = (front_face_score_np - front_face_score_np.min()) / (front_face_score_np.max() - front_face_score_np.min())
    front_face_score_np = front_face_score_np * 0.1 + 0.9
    face_scores_np = np.array(face_scores)
    access_score_np = front_face_score_np + face_scores_np

    best_face_score_idx = np.argmax(access_score_np)
    best_idx = face_idx_candis[best_face_score_idx]
    # for i in range(len(face_scores)):
    #     print(face_idx_candis[i], face_scores[i])
    # print(best_idx)
    return best_idx


def show_card_detect(candis, cap, values):
    rect_args = load_model()
    for candi in candis:
        # k = 20 if 20 < len(candi) else len(candi)
        # idxs = topk_idx(candi, k) + candi[0]
        idxs = candi
        idxs = pick_up_candi(candi, 20, values)

        for idx in idxs:
            cap.set(cv.CAP_PROP_POS_FRAMES, START_FRAME + idx)

            res, frame = cap.read()
            frame = frame[250: -250, 500:-500]

            print(START_FRAME + idx)
            croped_img, rect_score = id_card_detect(frame, rect_args)
            if not croped_img.width:

                cv.imshow('imgae', put_text(frame, str(rect_score)))
                if cv.waitKey(1) == ord('q'):
                    break
            else:
                croped_img = cv.cvtColor(np.asarray(croped_img), cv.COLOR_RGB2BGR)
                half = croped_img[int(croped_img.shape[0] * 0.1): int(croped_img.shape[0] * 0.7),
                       int(croped_img.shape[1] * 0.55): int(croped_img.shape[1] * 0.95)]
                half_eye = retina_face_detect(half)
                cv.imshow('imgae', put_text(croped_img, str(rect_score)))
                if cv.waitKey(1) == ord('q'):
                    break
        cv.waitKey(0)


def get_img(cap, frame_num):
    cap.set(cv.CAP_PROP_POS_FRAMES, frame_num)
    s_f, frame = cap.read()
    if s_f:
        return frame
    else:
        return None


if __name__ == '__main__':
    num = '2'
    height, width, frames_num, frames_per_sec, cap = get_video_info_and_header('test_samples/' + num + '.avi')
    height, width, frames_num, frames_per_sec = map(int, (height, width, frames_num, frames_per_sec))
    START_FRAME = int(frames_per_sec * 15)
    print(height, width, frames_num, frames_per_sec)

    pix_value_average = get_average_pixel_value(cap)
    # np.save('hsv_values_1.npy', pix_value_average)
    # pix_value_average = np.load('hsv_values_' + num + '.npy')

    # plt.plot([x for x in range(len(pix_value_average))], pix_value_average)
    # plt.show()
    # plt.pause(0.5)
    paper_candis, id_candis = find_candidates(pix_value_average)
    # show_candidates(id_candis, cap)
    # show_card_detect(id_candis, cap, pix_value_average)

    people_face_idx = find_best_face(frames_num - START_FRAME, cap)
    people_face_img = get_img(cap, START_FRAME + people_face_idx)
    cv.imwrite('./output/' + num + '_face.png', people_face_img)

    id_card_idx, id_card_idx_candis = find_id_card(id_candis, cap, pix_value_average)
    id_card_img = get_img(cap, START_FRAME + id_card_idx)
    cv.imwrite('./output/' + num + '_id_card.png', id_card_img)

    paper_idx = find_paper(paper_candis, id_card_idx, pix_value_average)
    paper_img = get_img(cap, START_FRAME + paper_idx)
    cv.imwrite('./output/' + num + '_contract.png', paper_img)



    # save result


    # show result
    # cv.imshow('1', paper_img)
    # cv.waitKey(0)
    # cv.imshow('1', id_card_img)
    # cv.waitKey(0)
    # cv.imshow('1', people_face_img)
    # cv.waitKey(0)

    cap.release()
    cv.destroyAllWindows()

    import dlib
