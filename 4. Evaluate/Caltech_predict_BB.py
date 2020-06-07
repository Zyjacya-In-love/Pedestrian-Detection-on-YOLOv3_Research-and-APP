import os

import cv2
import numpy as np
from PIL import Image
from timeit import default_timer as timer

from keras_yolov3_detect import YOLO_detector



'''
对于 Caltech 测试数据集 ./image，使用 keras_yolov3_detect.py 预测出每张图片的 bounding boxes 信息，
对于每张图片的每个目标，其预测边界框由 4 个 int 表达，分别是 [left, top, width, height]
PS：一行 存储一张图片中的 所有 边界框坐标，Test：10000 行
'''


def solve(detector):
    original_image_path = './code/data-USA/images/'
    save_path = "./code/data-USA/res/YOLOv3/"
    second_name = 'sec.npy'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    take_seconds = []
    cnt = 0

    for root, sub_dir, files in os.walk(original_image_path):
        now_save_path = save_path + '/'.join(root.split('/')[4:])
        if not os.path.exists(now_save_path):
            os.makedirs(now_save_path)
        for file in files:
            image_path = os.path.join(root, file)
            img = Image.open(image_path)
            start = timer()
            predict_bb, predict_scores = detector.detect_image(img)
            end = timer()
            take_seconds.append(end - start)
            predict_txt_file = open(now_save_path + '/' + '%s.txt' % (file.split('.')[0]), 'w') # 写预测txt
            # print("predict_bb : ", predict_bb)
            for i, bbox in enumerate(predict_bb):
                x1, y1, x2, y2 = bbox
                bbox = (x1, y1, x2-x1, y2-y1, predict_scores[i])
                # print(",".join([str(a) for a in bbox]))
                predict_txt_file.write(",".join([str(a) for a in bbox]) + '\n')
            cnt += 1

    take_seconds = np.array(take_seconds)
    # np.save(save_path + second_name, take_seconds)
    avg_sec = np.mean(take_seconds)
    print("avg_sec : ", avg_sec)



if __name__ == '__main__':
    detector = YOLO_detector(score=0)
    solve(detector)