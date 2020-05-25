import os
import shutil
import numpy as np
import re

'''
制作 训练 测试 需要的原始数据 和 标签
正样本：902
负样本：1617
/pos：Train：614 个，Test：288 个
/neg：Train：1218 个，Test：453 个
1. 数据
将 ./INRIAPerson /Train 和 /Test 中 的 /pos /neg 都 拷贝 到一个文件夹 ./data/
文件结构如下：
./data
    /train -- 1832
    /test -- 741
2. 标签
image_path xmin,ymin,xmax,ymax,class_id xmin,ymin,xmax,ymax,class_id
'''


# sets=['Train', 'Test']
sets=[('Train', 'pos'), ('Train', 'neg'), ('Test', 'pos'), ('Test', 'neg')]


classes = ["person"]


def convert_annotation(image_set, image_id, list_file):
    in_file = './INRIA/INRIAPerson/%s/annotations/%s.txt'%(image_set, image_id)
    with open(in_file, 'rb') as f:
        ann = f.read()
    Ground_Truth_list = re.findall('\(\d+, \d+\)[\s\-]+\(\d+, \d+\)', str(ann))  # like (250, 151) - (299, 294)
    coordinates = [re.findall('\d+', box) for box in Ground_Truth_list]
    coordinates = np.array(coordinates, dtype='int')
    for obj in coordinates:
        list_file.write(" " + ",".join([str(a) for a in obj]) + ',' + str(0))


if __name__ == '__main__':

    for image_set, pORn in sets:
        images = open('./INRIA/INRIAPerson/%s/%s.lst'%(image_set, pORn)).read().strip().split()
        testOrTrain = 'test' if 'Test'==image_set else 'train'
        list_file = open('%s.txt'%(testOrTrain), 'a') # ann.txt 说明txt文件
        save_path = "./data/{}/".format(testOrTrain) # 图片保存目录
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for file in images:
            img_path = './INRIA/INRIAPerson/'
            image_id = file.split('/')[-1].split('.')[0]
            src = img_path + file
            des = save_path + file.split('/')[-1]

            # 复制图片
            shutil.copyfile(src, des)

            # 写 annotation
            list_file.write(des)
            if pORn == 'pos':
                convert_annotation(image_set, image_id, list_file)
            list_file.write('\n')


        list_file.close()

