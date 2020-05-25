from pycocotools.coco import COCO
import os
import shutil
import numpy as np
import re
import json
from collections import defaultdict

'''
制作 训练 测试 需要的原始数据 和 标签
正样本：2693 + 64115 = 66808
负样本：2307 + 54172 = 56479

1. 数据
将 ./COCO /train 和 /val 都 拷贝 到一个文件夹 ./data/train
文件结构如下：
/data
    /train -- 118287 + 5000 = 123287
2. 标签
image_path xmin,ymin,xmax,ymax,class_id xmin,ymin,xmax,ymax,class_id
'''


# sets=['Train', 'Test']
# sets=['train2017', 'val2017']
sets=[('2017', 'train'), ('2017', 'val')]
# sets=[('2017', 'train')]

classes = ["person"]


def convert_annotation(imgId, catId, list_file):
    # 边界框
    annIds = coco.getAnnIds(imgIds=imgId, catIds=catId, iscrowd=None)
    if len(annIds) > 0:
        anns = coco.loadAnns(annIds)
        for ann in anns:
            x, y, w, h = ann['bbox']
            x_min = int(x)
            y_min = int(y)
            x_max = x_min + int(w)
            y_max = y_min + int(h)
            bbox = (x_min, y_min, x_max, y_max)
            cls_id = catId.index(ann['category_id'])
            list_file.write(" " + ",".join([str(a) for a in bbox]) + ',' + str(cls_id))
        return True
    else :
        return False

if __name__ == '__main__':
    train_pos = 0
    train_neg = 0
    test_pos = 0
    test_neg = 0

    for year, image_set in sets:
        testOrTrain = 'train' if 'train'==image_set else 'test'
        list_file = open('%s.txt' % (testOrTrain), 'a')  # ann.txt 说明txt文件
        save_path = "./data/{}/".format(testOrTrain)  # 图片保存目录
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        dataDir = './COCO'
        dataType = '%s%s'%(image_set, year) # 'val2017'
        annFile = '{}/annotations/instances_{}.json'.format(dataDir, dataType)

        # initialize COCO api for instance annotations
        coco = COCO(annFile)

        # 知道 person 类的id是多少
        catId = coco.getCatIds(catNms=classes)
        # 所有类的图片的id
        imgIds = coco.getImgIds()

        for imgId in imgIds:
            img_path = '{}/{}/'.format(dataDir, dataType)
            file = '%012d.jpg' % imgId
            src = img_path + file
            des = save_path + file

            # 复制图片
            shutil.copyfile(src, des)

            # 写 annotation
            list_file.write(des)
            if convert_annotation(imgId, catId, list_file):
                if testOrTrain == 'test':
                    test_pos += 1
                else:
                    train_pos += 1
            else:
                if testOrTrain == 'test':
                    test_neg += 1
                else:
                    train_neg += 1
            list_file.write('\n')

        list_file.close()

    print("train_pos : ", train_pos)
    print("train_neg : ", train_neg)
    print("test_pos : ", test_pos)
    print("test_neg : ", test_neg)

