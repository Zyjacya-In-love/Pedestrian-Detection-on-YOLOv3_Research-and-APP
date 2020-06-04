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

For **./"2. Data"**, script are prepared individually by dataset. And the required directory structure :open_file_folder: is shown below before run python script.

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

:white_check_mark: **almost full copy from [keras-yolo3@qqwweee](https://github.com/qqwweee/keras-yolo3)**

By the way, `job-batch32-95gpu4.sh` is the script to submit job for training model. Just as a souvenir, do not care :)

### 3.1 Network & Loss

Here is the YOLOv3 network structure I draw for the project (by visio).

<img src="./__READMEimages__/YOLOv3_network.png">

According to the codes+papers+blogs+experiments, I also conclude a loss function (for reference only).

In fact, the total loss is equal to the sum of the loss values of the 3 layer output feature map. This is the loss function of the output feature map of any layer.

<p align=center>
    <img src="./__READMEimages__/loss_func.png" alt="loss" height="300">
</p>


### 3.2 Get Anchor

core copy from [keras-yolo3/kmeans.py@qqwweee](https://github.com/qqwweee/keras-yolo3/blob/master/kmeans.py), just ADD **visualization**

Before get anchors, that require train dataset Ground-Truth file (default `train.txt`) from [2\.4 Batch processing](#24-batch-processing)

```bash
python anchor_kmeans.py
```

After run script, the directory structure that will be **added** in the **3. Train** folder is as follows

```
+
├─Anchors-by-Kmeans.png
├─yolo_anchors.txt
└─kmeans.npy
```

`Anchors-by-Kmeans.png` is the visualization picture file.
`yolo_anchors.txt` is exactly the file to store the anchors needed for training.
`kmeans.npy` stores the Kmeans result, it helps to avoid repeated Kmeans when changing the visualization style.


Here is one of the anchors with relatively high accuracy, which is 65.81%, and it's also the train anchors I used. (Accuracy is the average value of IOU of each boundary-box and clustering center. The larger it is, the better the clustering effect of k-means will be.)
```
Anchors: (6,11), (12,22), (19,40), (33,65), (49,132), (68,81), (94,189), (170,287), (335,389)
```

<img src="./__READMEimages__/Anchors-by-Kmeans.png" height="330">

### 3.3 Pretrained Weights

follow [keras-yolo3@qqwweee/README.md](https://github.com/qqwweee/keras-yolo3/blob/master/README.md)

1\. Download YOLOv3 pre-trained weight file [here (237 MB)](https://pjreddie.com/media/files/yolov3.weights) from [YOLO website](https://pjreddie.com/darknet/yolo/). Or just run this:
```bash
wget https://pjreddie.com/media/files/yolov3.weights
```
2\. Convert the Darknet YOLO model(\*.cfg+\*.weights) to a Keras model(\*.h5).
```bash
python convert_yolov3_weight_darkent2keras.py -w yolov3.cfg yolov3.weights yolo_weights.h5
```

### 3.4 Prepare files needed for training

1\. create class file(`person_classes.txt`), each line is a class name, so just fill in one line: **person**.
```bash
vi person_classes.txt
```
2\. make a floder (`model_data`) place configuration file and pretrained weights.
```bash
mkdir model_data
```
3\. move  `person_classes.txt`, `yolo_anchors.txt` and `yolo_weights.h5` to `model_data`.
```bash
mv person_classes.txt yolo_anchors.txt yolo_weights.h5 ./model_data
```

### 3.5 True Train process

Before training, don't forget move train dataset (`/data`) and image_path file (`train.txt`) from [2\.4 Batch processing](#24-batch-processing) to  current directory (`3. Train`)

And then use [keras-yolo3@qqwweee/train.py](https://github.com/qqwweee/keras-yolo3/blob/master/train.py)'s default Train config, just save a weights per 10 epoch.

For train&val, All training data(`train.txt`) are shuffled randomly, and divide train:val = 9:1.

For 1 training stage, Train with frozen layers first, to get a stable loss.

1. freeze all but 3 output layers
2. optimizer=Adam()
3. learning_rate=1e-3
4. epochs=50

For 2 training stage, Unfreeze and continue training, to fine-tune.

1. Unfreeze all of the layers
2. optimizer=Adam()
3. learning_rate=1e-4
4. epochs=300-50
5. reduce_lr: When val_loss does not decrease for 3 epoch, the learning rate is reduced by 10% each time.
6. early_stopping: When val_loss does not decrease for 10 epochs, the training is stopped early.

Now we can finally train !!!
```bash
python train.py
```

For the model I trained, I push it on `pan.baidu.com` and `drive.google.com`, it's a weight that has been trained 109 epochs.(235.4MB)

Download link:

1. Baidu Cloud Disk(中文) : [https://pan.baidu.com/s/130J8c5B9RQILSDKAFKyK1A](https://pan.baidu.com/s/130J8c5B9RQILSDKAFKyK1A) password: wmht

2. Google Drive : [https://drive.google.com/file/d/1zxcpjklHKQ6hULa8HmeKIQqnkH_h_NSI/view?usp=sharing](https://drive.google.com/file/d/1zxcpjklHKQ6hULa8HmeKIQqnkH_h_NSI/view?usp=sharing)

### 3.6 Loss curve plot

The log was stored through the TensorBoard while training. So, just extract the log of loss and draw the loss curve.

The first parameter is the path to log of the first 50 epochs. The second parameter is the path to the rest of log after 50 epochs.

**NOTE:** Pay attention to the order

```bash
python Tensorboard_log_loss_visualization.py \
    logs/batch32/events.out.tfevents.1586104689.r1cmpsrvs79-14ig0702 \
    logs/batch32/events.out.tfevents.1586671921.r1cmpsrvs79-14ig0702
```

Here is the loss curve plot I trained model, it's 1 to 109 because it stopped early after 109 epoch. Finally, loss=9.05, val_loss=9.54.

<img src="./__READMEimages__/loss_curves.png" height="370">



## 4. Model Evaluation

## 5. Web App



## 6. Summary