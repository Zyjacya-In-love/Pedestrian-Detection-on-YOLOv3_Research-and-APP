import xml.etree.ElementTree as ET
from os import getcwd
import os
import shutil

# sets=[('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
# sets=[('2007', 'train')]
sets=[('2007', 'trainval'), ('2007', 'test'), ('2012', 'trainval')]

# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
classes = ["person"]


def convert_annotation(year, image_id, list_file):
    in_file = open('./VOC/VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    flag = False
    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult)==1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
        flag = True

    return flag


if __name__ == '__main__':
    wd = getcwd()
    train_pos = 0
    train_neg = 0
    test_pos = 0
    test_neg = 0

    for year, image_set in sets:
        image_ids = open('./VOC/VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
        # list_file = open('%s_%s.txt'%(year, image_set), 'w')
        testOrTrain = 'test' if 'test'==image_set else 'train'
        list_file = open('%s.txt'%(testOrTrain), 'a') # ann.txt 说明txt文件
        save_path = "./data/{}/".format(testOrTrain) # 图片保存目录
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for image_id in image_ids:
            img_path = './VOC/VOCdevkit/VOC%s/JPEGImages/' % (year)
            file = '%s.jpg'%(image_id)
            src = img_path + file
            des = save_path + file

            shutil.copyfile(src, des) # 图片复制完毕

            # list_file.write('%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(wd, year, image_id))
            list_file.write(des)
            if convert_annotation(year, image_id, list_file):
                if testOrTrain == 'test':
                    test_pos += 1
                else :
                    train_pos += 1
            else :
                if testOrTrain == 'test':
                    test_neg += 1
                else :
                    train_neg += 1
            list_file.write('\n')

        list_file.close()

    print("train_pos : ", train_pos)
    print("train_neg : ", train_neg)
    print("test_pos : ", test_pos)
    print("test_neg : ", test_neg)

