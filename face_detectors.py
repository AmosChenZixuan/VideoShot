from retinaface.dnn_detector import RetinafaceDetector_dnn
import cv2 as cv
import numpy as np

detector = RetinafaceDetector_dnn()


def retina_face_detect(img):
    what, landmarks, score = detector.detect_faces(img)
    if landmarks.size:
        landmarks = landmarks[0].astype(np.int)
        for i in range(5):
            cv.circle(img, (landmarks[i], landmarks[i + 5]), 10, (0, 0, 255), 0)
    return img


def retina_face_distinguish(img):
    what, landmarks, score = detector.detect_faces(img)
    status = (1, score, landmarks) if score.size else (0, 0, 0)
    return status


if __name__ == '__main__':
    ori_img = cv.imread('/home/lyz/Pictures/cy.jpeg')
    faces = retina_face_detect(ori_img)
    cv.imshow('1', faces)
    cv.waitKey(0)
