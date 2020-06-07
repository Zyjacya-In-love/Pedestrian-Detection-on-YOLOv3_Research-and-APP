import os

import cv2
import numpy as np
from PIL import Image
from timeit import default_timer as timer

from keras_yolov3_detect import YOLO_detector



'''
用于 做出 工具 Object-Detection-Metrics [https://github.com/rafaelpadilla/Object-Detection-Metrics#optional-arguments]
要求的格式的数据
测试集 评估 共 10693 张图片 
对于 Object-Detection-Metrics 工具要求 每张图片一个文件，文件名是：“图片名.txt”
真实框 ground truth files 其格式为
    <class_name> <left> <top> <right> <bottom>
检测出的作为 detection files 其格式为 
    <class_name> <confidence> <left> <top> <right> <bottom> 
'''


# ---------------------------------------------------------------------------------
# 一些设置
bb_save_path = "./PRcurve_AP_raw_bb_data"
if not os.path.exists(bb_save_path):
    os.makedirs(bb_save_path)
Ground_Truth_save_path = bb_save_path + '/gt'
predict_bb_save_path = bb_save_path + "/pre"
if not os.path.exists(Ground_Truth_save_path):
    os.makedirs(Ground_Truth_save_path)
if not os.path.exists(predict_bb_save_path):
    os.makedirs(predict_bb_save_path)

annotation_path = './test.txt'
# ---------------------------------------------------------------------------------

def get_Ground_Truth():
    print("get_Ground_Truth : ")
    with open(annotation_path) as f:
        lines = f.readlines()
    for line in lines:
        line = line.split()
        # map()函数将一个全部为str的列表，转化为全部为int的列表 list(map(int, box.split(',')[0:4]))
        gt_box = np.array([np.array(list(map(int, box.split(',')[0:4]))) for box in line[1:]])

        image_path = line[0]
        image_name = image_path.split('/')[-1].split('.')[0]

        gt_txt_file = open(Ground_Truth_save_path + '/' + '%s.txt' % (image_name), 'w')  # 写预测txt

        for i, bbox in enumerate(gt_box):
            gt_txt_file.write("person " + " ".join([str(a) for a in bbox]) + '\n')

        gt_txt_file.close()

    print(Ground_Truth_save_path, " done")


def get_predict_bb():
    print("get_predict_bb : ")
    detector = YOLO_detector(score=0)
    with open(annotation_path) as f:
        lines = f.readlines()

    take_seconds = []

    for line in lines:
        image_path = line.split()[0]
        img = Image.open(image_path)
        start = timer()
        predict_bb, predict_scores = detector.detect_image(img)
        end = timer()
        take_seconds.append(end - start)
        image_name = image_path.split('/')[-1].split('.')[0]
        predict_txt_file = open(predict_bb_save_path + '/' + '%s.txt'%(image_name), 'w')  # 写预测txt
        for i, bbox in enumerate(predict_bb):
            x1, y1, x2, y2 = bbox
            bbox = (predict_scores[i], x1, y1, x2, y2)
            predict_txt_file.write("person " + " ".join([str(a) for a in bbox]) + '\n')
        predict_txt_file.close()

    take_seconds = np.array(take_seconds)
    avg_sec = np.mean(take_seconds)
    print("predict ", np.shape(take_seconds)[0], " images!!!")
    print("avg_sec_/_img : ", avg_sec)
    print("max_sec_/_img : ", np.max(take_seconds))
    print("min_sec_/_img : ", np.min(take_seconds))

    print(predict_bb_save_path, " done")


if __name__ == '__main__':
    get_Ground_Truth()
    get_predict_bb()
