import numpy as np
from retinaface.prior_box import cfg_mnet, PriorBox_np
from retinaface.py_cpu_nms import py_cpu_nms
import cv2


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    # print('Missing keys:{}'.format(len(missing_keys)))
    # print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    # print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    # print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


class RetinafaceDetector_dnn:
    def __init__(self, model_path='retinaface/FaceDetector_320.onnx'):
        self.model = cv2.dnn.readNetFromONNX(model_path)
        self.im_height = int(model_path[:-5].split('_')[-1])
        self.im_width = int(model_path[:-5].split('_')[-1])
        priorbox = PriorBox_np(cfg_mnet, image_size=(self.im_height, self.im_width))
        self.prior_data = priorbox.forward()  ####PriorBox生成的一堆anchor在强项推理过程中始终是常数是不变量，因此只需要在构造函数里定义一次即可
        self.scale = np.array([[self.im_width, self.im_height]])

    #####使用numpy做后处理, 摆脱对pytorch的依赖
    def decode(self, loc, priors, variances):
        boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                                priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    def decode_landm(self, pre, priors, variances):
        landms = np.concatenate((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                                 priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                                 priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                                 priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                                 priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                                 ), axis=1)
        return landms

    def detect_faces(self, img_raw, confidence_threshold=0.9, top_k=5, nms_threshold=0.4, keep_top_k=5, resize=1):
        blob = cv2.dnn.blobFromImage(img_raw, size=(self.im_width, self.im_height), mean=(104, 117, 123))
        self.model.setInput(blob)
        loc, conf, landms = self.model.forward(['loc', 'conf', 'landms'])

        boxes = self.decode(np.squeeze(loc, axis=0), self.prior_data, cfg_mnet['variance'])
        boxes = boxes * np.tile(self.scale, (1, 2)) / resize  ####广播法则
        scores = np.squeeze(conf, axis=0)[:, 1]
        landms = self.decode_landm(np.squeeze(landms, axis=0), self.prior_data, cfg_mnet['variance'])
        landms = landms * np.tile(self.scale, (1, 5)) / resize  ####广播法则

        # ignore low scores
        inds = np.where(scores > confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]
        scores = scores[keep]

        # keep top-K faster NMS
        dets = dets[:keep_top_k, :]
        landms = landms[:keep_top_k, :]
        # print(landms.shape)
        landms = landms.reshape((-1, 5, 2))
        # print(landms.shape)
        landms = landms.transpose((0, 2, 1))
        # print(landms.shape)
        landms = landms.reshape(-1, 10, )
        # print(landms.shape)
        srcim_scale = np.array([[img_raw.shape[1], img_raw.shape[0]]]) / self.scale
        dets[:, :4] = dets[:, :4] * np.tile(srcim_scale, (1, 2))  ###还原到原图上
        # landms = landms * np.tile(srcim_scale, (1, 5))    ####5个关键点坐标是x1,y1,x2,y2,x3,y3,x4,y4,x5,y5排列
        landms = landms * np.repeat(srcim_scale, 5, axis=1)  ####5个关键点坐标是x1,x2,x3,x4,x5,y1,y2,y3,y4,y5排列
        return dets, landms, scores

