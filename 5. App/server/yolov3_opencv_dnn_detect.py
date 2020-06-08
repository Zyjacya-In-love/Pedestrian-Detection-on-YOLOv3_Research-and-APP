import numpy as np
import argparse
import time
import cv2
import os

'''
提供边界框检测函数 YOLO_detector
借助 opencv 应用 yolo 网络进行目标检测
代码改编自：https://www.pyimagesearch.com/2018/11/12/yolo-object-detection-with-opencv/
需要的参数有：
image_path 待检测图像的位置
YOLO_path 是一个 元组
    0：configPath YOLO 网络配置文件的位置：'yolo-inria.cfg'
    1：weightsPath YOLO 网络训练好的参数权重的文件位置 'yolo-inria.weights'
min_confidence=0.3 最小置信值，用于过滤比较小概率的检测框
nms_threshold=0.45 NMS 阈值
'''

class YOLO_detector():
    _defaults = {
        "config_path": 'model/yolo-person.cfg',
        "weights_path": 'model/yolov3-keras2darknet.weights',
        "min_confidence": 0.3,
        "nms_threshold": 0.45,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)  # set up default values
        self.__dict__.update(kwargs)  # and update with user overrides
        self.net = cv2.dnn.readNetFromDarknet(self.config_path, self.weights_path)

    def detect_image(self, image):
        (H, W) = image.shape[:2]
        # determine only the *output* layer names that we need from YOLO
        ln = self.net.getLayerNames()
        ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        # construct a blob from the input image
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        # perform a forward pass of the YOLO object detector, giving us our bounding boxes and associated probabilities
        self.net.setInput(blob)
        layerOutputs = self.net.forward(ln)
    # initialize our lists of detected bounding boxes, confidences, and
        boxes = []
        confidences = []
        # print("dsadadasdsaasdddsad")
    # loop over each of the layer outputs
        for output in layerOutputs:
            # loop over each of the detections
            for detection in output:
                # 因为 class 只有 一个 person 所以 detection[5] 就是 preson 的 置信度
                confidence = detection[5]
                # filter out weak predictions by ensuring the detected
                # probability is greater than the minimum probability
                if confidence > self.min_confidence:
                    # scale the bounding box coordinates back relative to the
                    # size of the image, keeping in mind that YOLO actually
                    # returns the center (x, y)-coordinates of the bounding
                    # box followed by the boxes' width and height
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    # use the center (x, y)-coordinates to derive the top and
                    # and left corner of the bounding box
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    # update our list of bounding box coordinates, confidences,
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))

    # # apply non-maxima suppression to suppress weak, overlapping bounding boxes
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.min_confidence, self.nms_threshold)
    # print("image_path: ", image_path)
    # print(type(idxs))
    # 改成 统一格式 (Xmin, Ymin) - (Xmax, Ymax) 即可用于 opencv 画图，也可用于后期和 gt 比较
        idxs = np.array(idxs)
        # print(idxs.shape)
        if idxs.shape[0] == 0:
            return np.array([]), np.array([])
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        nms_boxes = boxes[idxs.flatten()]
        nms_confindences = confidences[idxs.flatten()]
        for i in range(len(nms_boxes)):
            (x, y) = (nms_boxes[i][0], nms_boxes[i][1])
            (w, h) = (nms_boxes[i][2], nms_boxes[i][3])
            nms_boxes[i][2] = x + w
            nms_boxes[i][3] = y + h
        # print(nms_boxes)
    # for this turn
        '''
        if len(nms_boxes) > 0:
            # loop over the indexes we are keeping
            for i in range(len(nms_boxes)):
                # extract the bounding box coordinates
                (x1, y1) = (nms_boxes[i][0], nms_boxes[i][1])
                (x2, y2) = (nms_boxes[i][2], nms_boxes[i][3])
                # draw a bounding box rectangle and label on the image
                # color = [int(c) for c in COLORS[classIDs[i]]]
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        '''
        # print(image)
        # print(type(image))
        # print(np.shape(image))
        # cv2.imshow('after', image)
        # cv2.waitKey()
        # return image
        return nms_boxes, nms_confindences
    # return boxes, nms_boxes


if __name__ == '__main__':
    image_path = './data/Test/person_032.png' # 没有行人
    # image_path = './data/Test/person_265.png' # 有行人
    YOLO_path = ('./cfg/yolo-inria.cfg', './backup/yolo-inria_130000.weights')

    pic, boxes = YOLO_detector(image_path, YOLO_path)
    # print(boxes)
    image = cv2.imread(image_path)
    # ensure at least one detection exists
    if len(boxes) > 0:
        # loop over the indexes we are keeping
        for i in range(len(boxes)):
            # extract the bounding box coordinates
            (x1, y1) = (boxes[i][0], boxes[i][1])
            (x2, y2) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            # color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
    # cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
    # 	0.5, color, 2)

    copy_im = image.copy()
    if len(pic) > 0:
        for i in range(len(pic)):
            (x, y) = (pic[i][0], pic[i][1])
            (w, h) = (pic[i][2], pic[i][3])
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

    # show the output image
    # cv2.imshow("Image", image)
    # cv2.waitKey(0)
    cv2.imwrite("before.png", copy_im)
    cv2.imwrite("after.png", image)
