import os
import shutil
from collections import namedtuple
import numpy as np
import cv2
from PIL import Image
from timeit import default_timer as timer

from yolov3_opencv_dnn_detect import YOLO_detector


'''
测试集 评估 共 10693 张图片 
    test.txt 中有真实框（Ground Truth） 存为 Ground_Truth.npy
    预测框需要现测（predict bb） 文件是 predict_bb.npy

评价 行人检测器 detector 的 检测质量（detection quality）
	主要是体现在输入图像中的行人能否被成功检测出来，以及得到的位置是否准确
使用 IoU 这个指标来评价 检测器 的准确率，
若 Ground Truth 与 predict bb 的 IOU score > 0.5 即认为这个窗口是正确的

PS： 
将 Ground Truth 与 predict bb 一同画在原始图像上并存储在 ./vs 中
./vs -- 10693
'''

# ---------------------------------------------------------------------------------
# 一些设置
IoUThresh = 0.5

bb_save_path = "./test_darknet-out-model_eval/"
if not os.path.exists(bb_save_path):
	os.makedirs(bb_save_path)
Ground_Truth_file = bb_save_path + 'Ground_Truth.npy'
predict_bb_file = bb_save_path + "predict_bb.npy"
pre_bb_second_file = bb_save_path + 'pre_bb_sec.npy'

vs_image_save_path = bb_save_path + "vs/"
error_image_save_path = bb_save_path + "error/"
missing_image_save_path = bb_save_path + "missing/"
correct_image_save_path = bb_save_path + "correct/"

error_missing_correct_txt_save_file = bb_save_path + "ErrorMissingCorrect.txt"

annotation_path = './test.txt'
# ---------------------------------------------------------------------------------

# define the `Detection` object
Detection = namedtuple("Detection", ["image_path", "gt", "pred"])

def bb_intersection_over_union(boxA, boxB):
	if boxA.shape[0] == 0 or boxB.shape[0] == 0:
		return 0
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)

	# return the intersection over union value
	return iou

def put_box2image(detection, save_path):
	# load the image
	image = cv2.imread(detection.image_path)
	# image = imread(original_image_path + detection.image_name)
	# draw the ground-truth bounding box along with the predicted bounding box
	for gt in detection.gt:
		cv2.rectangle(image, tuple(gt[:2]), tuple(gt[2:]), (0, 255, 0), 2) # green
	for pred in detection.pred:
		cv2.rectangle(image, tuple(pred[:2]), tuple(pred[2:]), (0, 0, 255), 2) # red
	# save the output image
	image_name = detection.image_path.split('/')[-1]
	cv2.imwrite(save_path+image_name, image)

def write_error_missing_txt_file(CorEorM, detection, this_pre, this_gt, this_good):
	txt_file = open(error_missing_correct_txt_save_file, 'a')
	txt_file.write("%s : %s\n" % (CorEorM, detection.image_path))
	txt_file.write("this_pre : %d\n" % this_pre)
	txt_file.write("this_gt : %d\n" % this_gt)
	txt_file.write("this_good : %d\n" % this_good)
	txt_file.close()



def evaluate(IsVsImg=False, IsErrorMissingCorrect=False):
	if IsVsImg and not os.path.exists(vs_image_save_path):
		os.makedirs(vs_image_save_path)
	else :
		IsVsImg = False
	if IsErrorMissingCorrect:
		if not os.path.exists(error_image_save_path):
			os.makedirs(error_image_save_path)
		else:
			IsErrorMissingCorrect = False
		if not os.path.exists(missing_image_save_path):
			os.makedirs(missing_image_save_path)
		else:
			IsErrorMissingCorrect = False
		if not os.path.exists(correct_image_save_path):
			os.makedirs(correct_image_save_path)
		else:
			IsErrorMissingCorrect = False
	print("IsVsImg : ", IsVsImg, "\nIsErrorMissingCorrect : ", IsErrorMissingCorrect)

	# 加载 gt、pred bb 数据，没有现算
	if not os.path.isfile(Ground_Truth_file):
		get_Ground_Truth()
	if not os.path.isfile(predict_bb_file):
		get_predict_bb()
	Ground_Truth = np.load(Ground_Truth_file, allow_pickle=True)
	predict_bb = np.load(predict_bb_file, allow_pickle=True)

	# 做一个Detection结构，方便计算两框 IOU
	with open(annotation_path) as f:
		lines = f.readlines()
	data = []
	for i in range(len(lines)):
		image_path = lines[i].split()[0]
		data.append(Detection(image_path, Ground_Truth[i], predict_bb[i]))

	pred_num = 0
	gt_num = 0
	good = 0
	if IsErrorMissingCorrect and os.path.isfile(error_missing_correct_txt_save_file):
		os.remove(error_missing_correct_txt_save_file)

	error_img_num = 0
	missing_img_num = 0
	exactly_correct_img_num = 0
	ErrorMissing_img_num = 0

	# loop over the example detections 循环所有图片
	for detection in data:
		this_good = 0
		this_pre = np.shape(detection.pred)[0]
		this_gt = np.shape(detection.gt)[0]

		pred_num += this_pre
		gt_num += this_gt

		for pred in detection.pred:
			for gt in detection.gt:
				iou = bb_intersection_over_union(gt, pred)
				if iou > IoUThresh:
					good += 1
					this_good += 1
					break

		if IsVsImg:
			put_box2image(detection, save_path=vs_image_save_path)

		# 完全正确
		IsThisCorrectImg = (this_pre == this_good and this_gt == this_good)
		if IsThisCorrectImg:
			exactly_correct_img_num += 1
		# 误检 Error
		IsThisErrorImg = (this_pre > this_gt or this_pre > this_good)
		if IsThisErrorImg:
			error_img_num += 1
		# 漏检 Missing
		IsThisMissingImg = (this_gt > this_pre  or this_gt > this_good)
		if IsThisMissingImg:
			missing_img_num += 1
		# 即 误检 又 漏检
		if IsThisErrorImg and IsThisMissingImg:
			ErrorMissing_img_num += 1

		if IsErrorMissingCorrect:
			if IsThisErrorImg:
				# 写 txt
				write_error_missing_txt_file("error detect", detection, this_pre, this_gt, this_good)
				# 写图片
				put_box2image(detection, save_path=error_image_save_path)
			if IsThisMissingImg:
				# 写 txt
				write_error_missing_txt_file("missing", detection, this_pre, this_gt, this_good)
				# 写图片
				put_box2image(detection, save_path=missing_image_save_path)
			if IsThisCorrectImg:
				# 写 txt
				write_error_missing_txt_file("correct!!!", detection, this_pre, this_gt, this_good)
				# 写图片
				put_box2image(detection, save_path=correct_image_save_path)

	# list_file.close()

	print("\n\n")
	print('------------------------------------------------------------')
	print('There are %d images.' % len(lines))
	print("exactly_correct_img_num : ", exactly_correct_img_num)
	print('error_img_num : ', error_img_num)
	print('missing_img_num : ', missing_img_num)
	print('ErrorMissing_img_num : ', ErrorMissing_img_num)
	print()
	print('Ground Truth sum number is', gt_num)
	print('predictions bb number is', pred_num)
	print('number of correct prediction is', good)
	print('Precision is', float(good * 1.0 / pred_num)) # 查准率（precision）=TP/(TP+FP)
	print('Recall is', float(good * 1.0 / gt_num)) # 查全率（Recall）=TP/(TP+FN)
	print('------------------------------------------------------------')
	print("\n\n")

def get_Ground_Truth():
	print("get_Ground_Truth : ")
	save_Ground_Truth = []
	with open(annotation_path) as f:
		lines = f.readlines()
	for line in lines:
		line = line.split()
		# map()函数将一个全部为str的列表，转化为全部为int的列表 list(map(int, box.split(',')[0:4]))
		box = np.array([np.array(list(map(int, box.split(',')[0:4]))) for box in line[1:]])
		save_Ground_Truth.append(box)
	save_Ground_Truth = np.array(save_Ground_Truth)
	np.save(Ground_Truth_file, save_Ground_Truth)
	che = np.load(Ground_Truth_file, allow_pickle=True)
	for i in range(che.shape[0]):
		if (che[i] == save_Ground_Truth[i]).all():
			continue
		else :
			print(i)
	print(Ground_Truth_file, " done and check pass")


def get_predict_bb():
	print("get_predict_bb : ")
	detector = YOLO_detector()
	with open(annotation_path) as f:
		lines = f.readlines()
	save_predict_bb = []
	take_seconds = []

	for line in lines:
		image_path = line.split()[0]
		img = cv2.imread(image_path)
		# img = Image.open(image_path)
		start = timer()
		predict_bb, _ = detector.detect_image(img)
		end = timer()
		take_seconds.append(end - start)
		save_predict_bb.append(predict_bb)

	take_seconds = np.array(take_seconds)
	np.save(pre_bb_second_file, take_seconds)
	avg_sec = np.mean(take_seconds)
	print("predict ", np.shape(take_seconds)[0], " images!!!")
	print("avg_sec_/_img : ", avg_sec)
	print("max_sec_/_img : ", np.max(take_seconds))
	print("min_sec_/_img : ", np.min(take_seconds))

	save_predict_bb = np.array(save_predict_bb)
	np.save(predict_bb_file, save_predict_bb)
	che = np.load(predict_bb_file, allow_pickle=True)
	for i in range(che.shape[0]):
		if (che[i] == save_predict_bb[i]).all():
			continue
		else :
			print("error: ", i)
	print(predict_bb_file, " done and check pass")


if __name__ == '__main__':
	evaluate(IsVsImg=True,IsErrorMissingCorrect=True)








