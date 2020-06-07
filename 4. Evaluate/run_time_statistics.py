import os
import matplotlib.pyplot as plt

import cv2
import numpy as np

import argparse


parser = argparse.ArgumentParser(description='Draw the run-time statistics.')
parser.add_argument('-p', '-pre_bb_sec_file', dest='pre_bb_sec', default='test_eval/pre_bb_sec.npy', help='Path to each image\'s seconds it takes to process prediction bounding-box')




def draw_histogram(images_seconds):
	x = np.arange(0.02, 0.11, 0.01)
	y = np.histogram(images_seconds, bins=x, range=None)
	# print(y)
	# 可视化数据
	figure, ax = plt.subplots(figsize=(19, 10))
	multiple = 2.5
	fontsize = 15 * multiple
	x_size = 50 * (multiple + 1)
	font = {'family': 'Times New Roman',
	        'weight': 'normal',
	        'size': fontsize, }
	plt.grid(linestyle="--")
	# 设置坐标刻度值的大小以及刻度值的字体
	plt.tick_params(labelsize=fontsize)
	labels = ax.get_xticklabels() + ax.get_yticklabels()
	[label.set_fontname('Times New Roman') for label in labels]

	label = ['%s-%s'%(str(int(y[1][i]*1001)),str(int(y[1][i+1]*1001))) for i in range(y[0].shape[0])]
	print(label)
	plt.xticks(range(y[0].shape[0]),label,rotation=0)
	plt.bar(range(y[0].shape[0]), y[0], align = 'center',color='steelblue', alpha = 0.8)
	for x,y in enumerate(y[0]):
	    plt.text(x,y+100,'%s' %round(y,1),font, ha='center')
	plt.xlabel('run time/image(ms)', font)
	plt.ylabel('image number', font)
	# plt.show()
	plt.savefig("runtime.png")  # 保存


def _main(args):
	# take_seconds = np.load("pre_bb_sec.npy", allow_pickle=True)
	take_seconds = np.load(args.pre_bb_sec, allow_pickle=True)
	# print(np.argmax(take_seconds))
	new_sec = take_seconds[1:]
	print("predict ", np.shape(take_seconds)[0], " images!!!")
	avg_sec = np.mean(new_sec)
	print("first : ", take_seconds[0])
	print("statistics ", np.shape(new_sec)[0], " images!!!")
	print("avg_sec_/_img : ", avg_sec)
	print("max_sec_/_img : ", np.max(new_sec))
	print("min_sec_/_img : ", np.min(new_sec))

	draw_histogram(new_sec)



if __name__ == '__main__':
    _main(parser.parse_args())