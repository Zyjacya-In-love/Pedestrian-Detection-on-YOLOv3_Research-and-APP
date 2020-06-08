# -*- coding: utf-8 -*- 
import argparse
import os

import numpy as np 
import keras 
from keras.models import load_model 
from keras.layers import Input

from yolo3.model import yolo_body


parser = argparse.ArgumentParser(description="qqwweee-Keras-yolo3's weights to model")
parser.add_argument('-h5', dest='h5_path', default='model/trained_weights_final.h5', help="Path to keras weight *.h5")
parser.add_argument('-cls', dest='classes_path', default='model/person_classes.txt', help="Path to class file")
parser.add_argument('-ahr', dest='anchors_path', default='model/yolo_anchors.txt', help="Path to anchors file")
parser.add_argument('-sf', dest='save_file', default='model/trained_model_final.h5', help='final save *.weights file')




def YOLOv3Model(input_shape, h5_path, class_names, anchors):
    # Load model, or construct model and load weights.
    # anyway, it's important for the class
    num_anchors = len(anchors)
    num_classes = len(class_names)
    try:
        yolo_model = load_model(h5_path, compile=False)
    except:
        yolo_model = yolo_body(Input(shape=(input_shape[0], input_shape[1], 3)), num_anchors // 3, num_classes)
        yolo_model.load_weights(h5_path)  # make sure model, anchors and classes match
    else:
        assert yolo_model.layers[-1].output_shape[-1] == \
               num_anchors / len(yolo_model.output) * (num_classes + 5), \
            'Mismatch between model and given anchor and class sizes'

    # 至此，Keras-YOLOv3 模型加载完毕
    print('{} model, anchors, and classes loaded.'.format(h5_path))
    return yolo_model


def _get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def _get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def _main(args):
    class_names = _get_class(args.classes_path)
    anchors = _get_anchors(args.anchors_path)

    input_shape = (416,416)
    model = YOLOv3Model(input_shape, args.h5_path, class_names, anchors)
    model.save(args.save_file)



if __name__ == "__main__":
    _main(parser.parse_args())