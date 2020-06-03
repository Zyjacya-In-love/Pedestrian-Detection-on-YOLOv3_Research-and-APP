<p align=center>
    <img src="./__READMEimages__/first_logo15.png" alt="Pedestrian Detection on YOLOv3 Research and APP">
</p>

<p align="center">
    Data+Train+Evaluate+App 4in1 repo within the paper
<a href='README-cn.md'>[中文版（TODO）]</a> <u><b>[English]</b></u>
</p>

<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/Zyjacya-In-love/Pedestrian-Detection-on-YOLOv3_Research-and-APP">
    <img src="https://img.shields.io/badge/repo%20size-37%20MB-blue" alt="Repo size">
    <img src="https://img.shields.io/badge/code%20size-11.83%20MB-blue" alt="Code size">
    <img src="https://img.shields.io/github/forks/Zyjacya-In-love/Pedestrian-Detection-on-YOLOv3_Research-and-APP?label=forks&style=social" alt="GitHub forks">
    <img src="https://img.shields.io/github/stars/Zyjacya-In-love/Pedestrian-Detection-on-YOLOv3_Research-and-APP?label=stars&style=social" alt="GitHub stars">
    <img src="https://img.shields.io/github/last-commit/Zyjacya-In-love/Pedestrian-Detection-on-YOLOv3_Research-and-APP?style=flat" alt="commit">
</p>

This is a repository that includes Pedestrian-Detection-on-YOLOv3_Research-and-APP, a 2020 undergraduate graduation project, **ALL codes**. The graduation project which has the **Data+Train+Evaluate+App 4in1 repo** Coded and paper Wrote by Ziqiang Xu from [Jiangnan University](https://www.jiangnan.edu.cn/).


### Table of Contents
- <a href='#1-Introduction'>1. Introduction</a>
- <a href='#2-Dataset'>2. Dataset</a>
- <a href='#3-YOLO-Train'>3. YOLO Train</a>
- <a href='#4-Model-Evaluation'>4. Model Evaluation</a>
- <a href='#5-Web-App'>5. Web App</a>
- <a href='#6-Summary'>6. Summary</a>


## 1. Introduction

**Pedestrian Detection** is a **subset** of **Object Detection** which only have one class of **person**. It aim to find out all pedestrians in the image or video's each frame, expressed location and size with **bounding-boxes**, just like this:

<img src="./__READMEimages__/pedestrian-detection-demo.BMP" height="200">

**YOLO (You Look Only Once)** is an advanced real-time object detection method. It is famous for processing images only once to get both location and classification, compared with previous object detection methods, while having similar accuracy with the state-of-the-art method, **YOLO run faster**.

This project researches Pedestrian Detection on YOLOv3 including **Data-convert,** **keras-Train**([keras-yolo3@qqwweee](https://github.com/qqwweee/keras-yolo3)) and **model-Evaluate**. Finally I also build a **Web App** base on **Flask** to realize the visualization of pedestrian detection results of the real-time webcam, image, or video (whose language is chinese, but you can easily use by following 5. Web App or just translating).


## 2. Dataset

### 2.1 Download
#### Microsoft COCO
**official link :** [http://cocodataset.org/](http://cocodataset.org/)

**Download link (2017 about 19GB) :**
```
http://images.cocodataset.org/zips/train2017.zip
http://images.cocodataset.org/zips/val2017.zip
http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```

annotation just use `instances_train2017.json`+`instances_val2017.json`

#### PASCAL VOC
**official link :** [http://host.robots.ox.ac.uk:8080/pascal/VOC/](http://host.robots.ox.ac.uk:8080/pascal/VOC/)

**Download link (07+12 about 2.7GB) :**
```
https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
```

after unzip just use `/JPEGImages`+`/Annotations`

**PS:** unzip `*.tar` file just like
```bash
tar xf *.tar
```

#### INRIA Person Dataset
~~**official link :** [http://pascal.inrialpes.fr/data/human/](http://pascal.inrialpes.fr/data/human/)~~ down

**Download link (about 969MB) :**

1. ~~official link : [http://pascal.inrialpes.fr/data/human/](http://pascal.inrialpes.fr/data/human/)~~ down

2. Baidu Cloud Disk(中文) : [https://pan.baidu.com/s/12TYw-8U9sxz9cUu2vxzvGQ](https://pan.baidu.com/s/12TYw-8U9sxz9cUu2vxzvGQ) password: jxqu

2. Google Drive : [https://drive.google.com/file/d/1wTxod2BhY_HUkEdDYRVSuw-nDuqrgCu7/view?usp=sharing](https://drive.google.com/file/d/1wTxod2BhY_HUkEdDYRVSuw-nDuqrgCu7/view?usp=sharing)

after unzip just use `/Train`+`/Test`



### 2.2 Data distribution

<img src="./__READMEimages__/data_table-en1-big.png" width="400">

**for train & test dataset divide**

> train = (COCO train) + (VOC 07+12 train+val) + (INRIA train)
>
> test = (COCO val) + (VOC 07 test) + (INRIA test)

<img src="./__READMEimages__/data_table-en2-big2.png">

### 2.3 Convert annotation format

These dataset's annotation format is not same, so convert Ground-Truth to a Uniform format.

The train and test set are stored as a TXT file respectively, One row for One image, just like the following (without [ ]):
```bash
[image_path] [bbox1] [bbox2] [bbox3] … [bboxn]
```
bbox format:
```bash
[x_min,y_min,x_max,y_max,0]
```
PS: 0 is class id，because only one class(person), 0 is enough

For example:
```txt
./data/train/000000391895.jpg 339,22,492,322,0 471,172,506,220,0
./data/train/000000309022.jpg
./data/test/000000397133.jpg 388,69,497,346,0 0,262,62,298,0
```

### 2.4 Batch processing

For **./"2. Data"**, script are prepared individually by dataset. And the required directory structure is shown below before run python script.

```
./"2. Data"
├─COCO
│  ├─annotations
│  ├─train2017
│  └─val2017
├─VOC
│  └─VOCdevkit
│      ├─VOC2007
│      └─VOC2012
├─INRIA
|  └─INRIAPerson
|      ├─Test
|      └─Train
├─coco_annotation.py
├─voc_annotation.py
└─INRIA_annotation.py
```

All python files are append write, so run separately is ok.

```bash
python coco_annotation.py
python voc_annotation.py
python INRIA_annotation.py
```

**PS:** the script will copy images to a **new folder** (./data/train or ./data/test)

After run script, the directory structure that will be **added** in the **2. Data** folder is as follows

```
+
├─data
│  ├─train
│  └─test
├─train.txt
└─test.txt
```


## 3. YOLO Train




## 4. Model Evaluation

## 5. Web App



## 6. Summary