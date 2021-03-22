import cv2 as cv
import pandas as pd
import numpy as np

from detect_id_card import id_card_detect
from helpers import *
from face_detectors import retina_face_detect, retina_face_distinguish


def pick_up_candi(candi, k, values):
    candi_values = values[candi]
    idx = lowk_idx(candi_values, k)
    return np.array(candi)[idx]


def put_text(img, text: str):
    font = cv.FONT_HERSHEY_SIMPLEX
    return cv.putText(img, text, (100, 100), font, 1.2, (255, 255, 255), 2)


def lowk_idx(v, k):
    pd_v = pd.Series(v)
    return pd_v.sort_values().index[:k].values.astype(np.int)


def get_video_info_and_header(path):
    cap = cv.VideoCapture(path)
    height, width, frames_num, frames_per_sec = cap.get(3), cap.get(4), cap.get(7), cap.get(5)
    return height, width, frames_num, frames_per_sec, cap


def get_hSv_value_and_extract_face(cap):
    num_frames = cap.get(7)
    pix_value_average = []
    interval = 15
    face_idx_candis = []
    face_scores = []
    front_face_score = []

    lm_test = []

    i = START_FRAME
    cap.set(cv.CAP_PROP_POS_FRAMES, i)
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            frame_roi = frame[int(frame.shape[0] * 0.1): int(frame.shape[0] * 0.9),
                        int(frame.shape[1] * 0.1): int(frame.shape[1] * 0.9)]

            hsv_img = cv.cvtColor(frame_roi, cv.COLOR_BGR2HSV)
            pix_value_average.append(hsv_img[:, :, 1].mean())

            # visualization
            imgzi = put_text(hsv_img, str(pix_value_average[-1]))
            cv.imshow('2', hsv_img)
            cv.waitKey(10)
            # end
            if i % interval == 0:
                has_face, score, lm = retina_face_distinguish(frame)
                # if has_face and score.shape[0] == 2:
                if has_face:
                    face_idx_candis.append(i)
                    face_scores.append(np.min(score))
                    front_face_score.append((lm[:, 1] - lm[:, 0]).mean())
                    lm_test.append(lm)

            if i >= num_frames - 30:  # end point
                break
            print(i)
        else:
            print('video cap ends at' + str(i) + 'th frame')
            break
        i += 1
    value_np = np.stack(pix_value_average, 0)
    cv.destroyAllWindows()

    if len(face_scores):

        front_face_score_np = np.array(front_face_score) + 0.01
        front_face_score_np = (front_face_score_np - front_face_score_np.min()) / (
                front_face_score_np.max() - front_face_score_np.min())
        front_face_score_np = front_face_score_np * 0.1 + 0.9
        face_scores_np = np.array(face_scores)
        access_score_np = front_face_score_np + face_scores_np

        best_face_score_idx = np.argmax(access_score_np)
        best_idx = face_idx_candis[best_face_score_idx]
        return value_np, best_idx
    else:
        return value_np, None


def clean_outsider(values):
    values = values[values < 200]
    values = values[values > 50]

    return values


def merge_close_list(lists):
    output_lists = []
    i = 0
    flag = 0
    while i < len(lists) - 1:
        if i == len(lists) - 2:
            flag = 1
        cur_list = lists[i]
        next_list = lists[i + 1]
        if next_list[0] - cur_list[-1] < 20:
            output_lists.append(list(range(cur_list[0], next_list[-1])))
            if flag:
                break
            i += 2
        else:
            output_lists.append(cur_list)
            if flag:
                output_lists.append(lists[-1])
            i += 1
    return output_lists


def cut(values, min_, med_, threshold_low, threshold_high, frame_length):
    low_value = min(min_, med_)
    high_value = max(min_, med_)
    tmp_list = []
    rs_list = []
    i = 0
    flag = 0
    while i < values.shape[0]:
        if low_value + (high_value - low_value) * threshold_low < values[i] < low_value + (
                high_value - low_value) * threshold_high:
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
    return merge_close_list(rs_list)


def cut_id_card(values, med_, threshold, frame_length):
    tmp_list = []
    rs_list = []
    i = 0
    flag = 0
    while i < values.shape[0]:
        if values[i] >= med_ * (1 + threshold) or values[i] <= med_ * (1 - threshold):
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
    return merge_close_list(rs_list)


def find_candidates(values):
    # mean_ = np.mean(pix_value_average)
    med = np.median(pix_value_average)
    min_ = np.min(pix_value_average)
    # mid = np.argmax(np.bincount(pix_value_average.astype(int)))

    print(med, min_)

    paper_res_list = cut(values, min_, med, 0, 0.65, 10)
    id_res_list = cut_id_card(values, med, 0.1, 10)

    return paper_res_list, id_res_list


def search_stable_img(best_candi, values):
    top_k_idx = pick_up_candi(best_candi, 15, values)
    dif = [(values[i - 2] + values[i + 2] - values[i] * 2) for i in top_k_idx]
    best_i = np.argmin(np.array(dif))
    opt_idx = top_k_idx[best_i]

    return opt_idx


def find_contract(paper_candis, start_frame, values):
    weight_list = []

    for idx, candi in enumerate(paper_candis):
        if candi[0] < start_frame or start_frame in candi:
            continue
        else:
            weight_list.append((idx, values[candi].mean() * 0.5 + 0.5 * len(candi)))

    if len(weight_list) == 0:
        raise ValueError('no contrast appears after idcard')
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
            frame = frame
            print(START_FRAME + idx)
            croped_img, rect_score = id_card_detect(frame, rect_args)

            if not croped_img.width:
                continue
            else:
                croped_img = cv.cvtColor(np.asarray(croped_img), cv.COLOR_RGB2BGR)
                half = croped_img[int(croped_img.shape[0] * 0.1): int(croped_img.shape[0] * 0.7),
                       int(croped_img.shape[1] * 0.55): int(croped_img.shape[1] * 0.95)]
                has_face, score, _ = retina_face_distinguish(half)
                cv.imshow('1', croped_img)
                cv.waitKey(5)

                if has_face:
                    id_card_index_candidate.append(idx)
                    face_scores.append(score)
    # cv.waitKey(0)
    if len(face_scores) == 0:
        raise ValueError('id_card_candidates wrong')

    best_face_score_idx = np.argmax(np.array(face_scores))
    best_idx = id_card_index_candidate[best_face_score_idx]

    return best_idx, id_card_index_candidate


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
            frame = frame

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


if __name__ == '__main__':
    import datetime
    starttime = datetime.datetime.now()

    video_path = './tmp_video/testwrite3.avi'
    video_name = video_path.split('/')[-1].split('.')[0]

    height, width, frames_num, frames_per_sec, cap = get_video_info_and_header(video_path)
    height, width, frames_num, frames_per_sec = map(int, (height, width, frames_num, frames_per_sec))
    # START_FRAME = frames_per_sec * 15
    START_FRAME = 0
    print(height, width, frames_num, frames_per_sec)

    pix_value_average, people_face_idx = get_hSv_value_and_extract_face(cap)
    people_face_img = get_img(cap, START_FRAME + people_face_idx)
    cv.imwrite('./output/' + video_name + '_face.png', people_face_img)
    # visual s values
    # import matplotlib.pyplot as plt
    # import matplotlib; matplotlib.use('TkAgg')
    # plt.plot([x for x in range(len(pix_value_average))], pix_value_average)
    # plt.show()
    # plt.pause(1000)
    # end

    contract_candis, id_candis = find_candidates(pix_value_average)
    if len(id_candis) == 0:
        raise ValueError('no id_card_candidates found, please check the video or adjust the threshold')
    if len(contract_candis) == 0:
        raise ValueError('no contract_candidates found, please check the video or adjust the threshold')

    # visual condidates
    # show_candidates(id_candis, cap)
    # show_candidates(contract_candis, cap)
    # show_card_detect(id_candis, cap, pix_value_average)

    # end

    id_card_idx, id_card_idx_candis = find_id_card(id_candis, cap, pix_value_average)
    id_card_img = get_img(cap, START_FRAME + id_card_idx)
    cv.imwrite('./output/' + video_name + '_id_card.png', id_card_img)

    paper_idx = find_contract(contract_candis, id_card_idx, pix_value_average)
    paper_img = get_img(cap, START_FRAME + paper_idx)
    cv.imwrite('./output/' + video_name + '_contract.png', paper_img)

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

    end_time = datetime.datetime.now()
    print("Running Time: ", (end_time - starttime).seconds)

    import dlib
