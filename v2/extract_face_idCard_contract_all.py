import argparse

import cv2 as cv
import pandas as pd
import numpy as np
import datetime

from detect_id_card import id_card_detect
from helpers import *
from face_detectors import retina_face_detect, retina_face_distinguish


parser = argparse.ArgumentParser(description='get input video path')
parser.add_argument('-v', '--video_path', type=str)

starttime = datetime.datetime.now()
rect_args = load_model()


def pick_up_candi(candi, k, values):
    """
    Find the whitest k frames from candidates
    """
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
    return map(int, (height, width, frames_num, frames_per_sec)), cap


def get_hSv_value_and_extract_face(cap, box0, box1, box2, box3):
    """
    Args:
        cap:  cv.VideoCapture
        box0: Area of Interest, x-left
        box1: Area of Interest, x-right
        box2: Area of Interest, y-left
        box3: Area of Interest, y-right

    Returns:
        value_np: [np.ndarray] pixel saturation average
        best_face_idx: [int] predicted best frame containing faces
        best_id_idx: [int] predicted best frame containing id cards
    """
    from config import INTERVAL, START_FRAME, N_FACE
    num_frames = cap.get(7)
    pix_value_average = []
    face_idx_candis = []
    face_scores = []
    front_face_score = []

    id_card_scores = []
    id_card_idxes = []

    # lm_test = []  # retina landmarks

    i = START_FRAME
    cap.set(cv.CAP_PROP_POS_FRAMES, i)
    while cap.isOpened():
        ret, frame = cap.read()
        # crop original frame
        frame = frame[box0: box1, box2: box3]
        if ret:
            # shrink down the region of interest
            frame_roi = frame[int(frame.shape[0] * 0.1): int(frame.shape[0] * 0.9),
                        int(frame.shape[1] * 0.1): int(frame.shape[1] * 0.9)]

            # get saturation (s-value) and save its mean value
            hsv_img = cv.cvtColor(frame_roi, cv.COLOR_BGR2HSV)
            pix_value_average.append(hsv_img[:, :, 1].mean())

            # detect using CNN
            vis_title = 'ori_pic'
            if i % INTERVAL == 0:
                # expected n faces. if detected, save the frame as a candidate
                has_face, score, lm = retina_face_distinguish(frame)
                if has_face and score.shape[0] == N_FACE:
                    vis_title += "-" + str(score.shape[0]) + " faces"
                    face_idx_candis.append(i)
                    face_scores.append(np.mean(score))
                    front_face_score.append((lm[:, 1] - lm[:, 0]).mean())
                    # lm_test.append(lm)

                # detect_id_card
                id_card_score = detect_id_card(frame)
                if id_card_score:
                    vis_title += "-with card"
                    id_card_scores.append(id_card_score)
                    id_card_idxes.append(i)
                print("Detecting ", i)
            # visualization
            # imgzi = put_text(hsv_img, str(pix_value_average[-1]))
            cv.imshow(vis_title, frame)
            cv.waitKey(1)
            # end
            if i >= num_frames - 30:  # end point
                break
        else:
            print('video cap ends at' + str(i) + 'th frame')
            break
        i += 1
    # from list to numpy
    value_np = np.stack(pix_value_average, 0)
    cv.destroyAllWindows()

    # find best faces and best id card, return their frame index
    if not len(face_scores) * len(id_card_scores):
        zero = 'FACE' if not len(face_scores) else 'IDCARD'
        raise ValueError(f'NO {zero} DETECTED. Modify the configurations and try again (see config.py).')
    else:
        front_face_score_np = np.array(front_face_score) + 0.01
        front_face_score_np = (front_face_score_np - front_face_score_np.min()) / (
                front_face_score_np.max() - front_face_score_np.min())
        front_face_score_np = front_face_score_np * 0.1 + 0.3
        face_scores_np = np.array(face_scores)
        access_score_np = front_face_score_np + face_scores_np * 0.01

        best_face_score_idx = int(np.argmax(access_score_np))
        best_face_idx = face_idx_candis[best_face_score_idx]

        best_id_card_score_idx = int(np.argmax(id_card_scores))
        best_id_idx = id_card_idxes[best_id_card_score_idx]
        return value_np, best_face_idx, best_id_idx

#
# def clean_outsider(values):
#     values = values[values < 200]
#     values = values[values > 50]
#
#     return values


def merge_close_list(lists):
    """
    1. merge consecutive lists into one single list if they are close to each other
    2. fill the middle indexes when merging
        e.g [[1,2,3], [6,7,8], [50,51,52]] -> [[1,2,3,4,5,6,7,8], [50,51,52]]
    """
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
    """
    looking for consecutive frames (last longer than frame_length)
    that has pixel values satisfying the threshold boundary

    Returns the merged array of frame sections (e.g. [[0-10], [50-60], ..]
    """
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
                    print('cut ', tmp_list[0], '-', tmp_list[-1])
                flag = 0
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

        print('cut id ', i)
        i += 1
    return merge_close_list(rs_list)


def find_candidates(values):
    from config import CUT_PAPER_ARGS
    # mean_ = np.mean(pix_value_average)
    med = np.median(pix_value_average)
    min_ = np.min(pix_value_average)
    # mid = np.argmax(np.bincount(pix_value_average.astype(int)))

    print(f'med:{med}, min:{min_}')

    paper_res_list = cut(values, min_, med, *CUT_PAPER_ARGS)
    # id_res_list = cut_id_card(values, med, 0.1, 10)
    id_res_list = paper_res_list  # Currently Not In Use

    return paper_res_list, id_res_list


def search_stable_img(best_candi, values):
    """
    Looking for the frame which has the lowest saturation (whiter),
        and has the least difference from its neighbor frames
    """
    top_k_idx = pick_up_candi(best_candi, 15, values)
    dif = [(values[i - 2] + values[i + 2] - values[i] * 2) for i in top_k_idx]
    best_i = np.argmin(np.array(dif))
    opt_idx = top_k_idx[best_i]

    return opt_idx


def find_contract(paper_candis, start_frame, values):
    """
    Args:
        paper_candis: [2D list] candidate frames sections which contain paper-like objects in it
        start_frame: [int] predicted best frame containing id cards.
            only looking for contracts after an id card is detected
        values: [np.ndarray] pixel saturation average

    Returns:
        paper_idx: [int] predicted best frame containing a contract.
    """
    weight_list = []

    for idx, candi in enumerate(paper_candis):
        if candi[0] < start_frame or start_frame in candi:
            # skip sections that happened before or during the id card displaying
            continue
        else:
            # calculate and save the weighted score
            weight_list.append((idx, values[candi].mean() * 0.5 + 0.5 * len(candi)))

    if len(weight_list) == 0:
        raise ValueError('no contrast appears after idcard')
    best_idx = int(np.argmax(np.array([x[-1] for x in weight_list])))
    best_candi = paper_candis[weight_list[best_idx][0]]
    paper_idx = search_stable_img(best_candi, values)
    return paper_idx


def detect_id_card(img):
    """
    looking for an id card from a frame, return the confidence score
    """
    croped_img, rect_score = id_card_detect(img, rect_args)
    if not croped_img.width:
        return 0
    else:
        # looking for a face in the right-middle part of the cropped area
        croped_img = cv.cvtColor(np.asarray(croped_img), cv.COLOR_RGB2BGR)
        half = croped_img[int(croped_img.shape[0] * 0.1): int(croped_img.shape[0] * 0.7),
               int(croped_img.shape[1] * 0.55): int(croped_img.shape[1] * 0.95)]
        has_face, score, _ = retina_face_distinguish(half)
        # cv.imshow('1', croped_img)
        # cv.waitKey(1)

        if has_face:
            return score
        else:
            return 0


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
            print('find id ', START_FRAME + idx)
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

            print('show card', START_FRAME + idx)
            croped_img, rect_score = id_card_detect(frame, rect_args)
            if not croped_img.width:

                cv.imshow('image', put_text(frame, str(rect_score)))
                if cv.waitKey(1) == ord('q'):
                    break
            else:
                croped_img = cv.cvtColor(np.asarray(croped_img), cv.COLOR_RGB2BGR)
                half = croped_img[int(croped_img.shape[0] * 0.1): int(croped_img.shape[0] * 0.7),
                       int(croped_img.shape[1] * 0.55): int(croped_img.shape[1] * 0.95)]
                half_eye = retina_face_detect(half)
                cv.imshow('image', put_text(croped_img, str(rect_score)))
                if cv.waitKey(1) == ord('q'):
                    break
        cv.waitKey(0)


def get_img(cap, frame_num):
    """
    get a particular frame from the cv.VideoCapture
    """
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
    from config import OUTPUT_FOLDER, START_FRAME, AOI
    try:
        args = parser.parse_args()
        video_path = args.video_path
        assert video_path is not None
    except AssertionError:
        video_path = 'test_samples/c3.avi'
    video_name = video_path.split('/')[-1].split('.')[0]
    write_root = OUTPUT_FOLDER + video_name

    vinfo, cap = get_video_info_and_header(video_path)
    height, width, frames_num, frames_per_sec = vinfo
    print(f"=========Video Info: {height}x{width}, {frames_num} frames, fps {frames_per_sec}=========")

    pix_value_average, people_face_idx, id_card_idx = get_hSv_value_and_extract_face(cap, *AOI)
    people_face_img = get_img(cap, START_FRAME + people_face_idx)
    cv.imwrite(write_root + '_face.png', people_face_img)

    id_card_img = get_img(cap, START_FRAME + id_card_idx)
    cv.imwrite(write_root + '_id_card.png', id_card_img)
    # visual s values
    # import matplotlib.pyplot as plt
    # import matplotlib; matplotlib.use('TkAgg')
    # plt.plot([x for x in range(len(pix_value_average))], pix_value_average)
    # plt.show()
    # plt.pause(1000)
    # end

    contract_candis, id_candis = find_candidates(pix_value_average)
    # if len(id_candis) == 0:
    #     raise ValueError('no id_card_candidates found, please check the video or adjust the threshold')
    if len(contract_candis) == 0:
        raise ValueError('no contract_candidates found, please check the video or adjust the threshold')

    # visual condidates
    # show_candidates(id_candis, cap)
    # show_candidates(contract_candis, cap)
    # show_card_detect(id_candis, cap, pix_value_average)

    # end

    # id_card_idx, id_card_idx_candis = find_id_card(id_candis, cap, pix_value_average)
    # id_card_img = get_img(cap, START_FRAME + id_card_idx)
    # cv.imwrite('./output/' + video_name + '_id_card.png', id_card_img)

    paper_idx = find_contract(contract_candis, id_card_idx, pix_value_average)
    paper_img = get_img(cap, START_FRAME + paper_idx)
    cv.imwrite(write_root + '_contract.png', paper_img)

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
