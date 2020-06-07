import cv2
import colorsys
import os
import traceback
from timeit import default_timer as timer

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from PIL import Image, ImageFont, ImageDraw

from yolo3.model import yolo_eval, yolo_body
from yolo3.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

"""
Keras-YOLOv3 行人检测 （边界框检测 bounding box） 
you can use detector to get bboxs for a image(PIL format)
"""

class YOLO_detector(object):
    _defaults = {
        "model_path": 'model/trained_weights_final.h5',
        "anchors_path": 'model/yolo_anchors.txt',
        "classes_path": 'model/person_classes.txt',
        "score" : 0.3,
        "iou" : 0.45,
        "model_image_size" : (416, 416),
        "gpu_num" : 1,
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        self.boxes, self.scores, self.classes = self.generate()

    # return [] include class names, for example: ['person']
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    # readline as str(because only one line) then split
    # return np.array include anchors whose shape (n, 2)
    # [[  6.  11.]
    #  [ 13.  25.]
    #  [ 24.  42.]
    #  [ 32.  73.]
    #  [ 55. 102.]
    #  [ 74. 195.]
    #  [120. 129.]
    #  [151. 280.]
    #  [309. 381.]]
    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    # 加载 Keras-YOLOv3 模型
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        # 断言终止，调试结束，为避免影响程序的性能，注释掉
        # assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        # anyway, it's important for the class
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        # 至此，Keras-YOLOv3 模型加载完毕
        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    # 提供的接口函数，用于接收图片PIL，返回行人检测的边界框结果
    def detect_image(self, image):
        # resize image if not Multiples of 416, 填充灰边 RGB(128,128,128)
        boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        # 归一化
        image_data = np.array(boxed_image, dtype='float32')
        image_data /= 255.
        # 变成批式数据 like (1, 416, 416, 3)
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
        # run
        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        if len(out_boxes) > 0:
            for i in range(len(out_boxes)):
                out_boxes[i][0], out_boxes[i][1] = swap(out_boxes[i][0], out_boxes[i][1])
                out_boxes[i][2], out_boxes[i][3] = swap(out_boxes[i][2], out_boxes[i][3])
        return out_boxes, out_scores
        # return out_boxes

    # close
    def close_session(self):
        self.sess.close()


def swap(t1, t2):
    return t2, t1
