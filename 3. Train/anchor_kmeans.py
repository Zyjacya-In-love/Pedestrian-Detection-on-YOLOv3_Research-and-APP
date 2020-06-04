import os

import numpy as np
import matplotlib.pyplot as plt

class YOLO_Kmeans:

    def __init__(self, cluster_number, filename, save_path):
        self.cluster_number = cluster_number
        self.filename = filename
        self.save_path = save_path

    def iou(self, boxes, clusters):  # 1 box -> k clusters
        n = boxes.shape[0]
        k = self.cluster_number

        box_area = boxes[:, 0] * boxes[:, 1]
        box_area = box_area.repeat(k)
        box_area = np.reshape(box_area, (n, k))

        cluster_area = clusters[:, 0] * clusters[:, 1]
        cluster_area = np.tile(cluster_area, [1, n])
        cluster_area = np.reshape(cluster_area, (n, k))

        box_w_matrix = np.reshape(boxes[:, 0].repeat(k), (n, k))
        cluster_w_matrix = np.reshape(np.tile(clusters[:, 0], (1, n)), (n, k))
        min_w_matrix = np.minimum(cluster_w_matrix, box_w_matrix)

        box_h_matrix = np.reshape(boxes[:, 1].repeat(k), (n, k))
        cluster_h_matrix = np.reshape(np.tile(clusters[:, 1], (1, n)), (n, k))
        min_h_matrix = np.minimum(cluster_h_matrix, box_h_matrix)
        inter_area = np.multiply(min_w_matrix, min_h_matrix)

        result = inter_area / (box_area + cluster_area - inter_area)
        return result

    def avg_iou(self, boxes, clusters):
        accuracy = np.mean([np.max(self.iou(boxes, clusters), axis=1)])
        return accuracy

    def kmeans(self, boxes, k, dist=np.median):
        box_number = boxes.shape[0]
        distances = np.empty((box_number, k))
        last_nearest = np.zeros((box_number,))
        np.random.seed()
        clusters = boxes[np.random.choice(
            box_number, k, replace=False)]  # init k clusters
        while True:

            distances = 1 - self.iou(boxes, clusters)

            current_nearest = np.argmin(distances, axis=1)
            if (last_nearest == current_nearest).all():
                break  # clusters won't change
            for cluster in range(k):
                clusters[cluster] = dist(  # update clusters
                    boxes[current_nearest == cluster], axis=0)

            last_nearest = current_nearest

        return clusters, current_nearest

    def result2txt(self, data):
        f = open("yolo_anchors.txt", 'w')
        row = np.shape(data)[0]
        for i in range(row):
            if i == 0:
                x_y = "%d,%d" % (data[i][0], data[i][1])
            else:
                x_y = ", %d,%d" % (data[i][0], data[i][1])
            f.write(x_y)
        f.close()

    def txt2boxes(self):
        f = open(self.filename, 'r')
        dataSet = []
        for line in f:
            infos = line.split(" ")
            length = len(infos)
            for i in range(1, length):
                width = int(infos[i].split(",")[2]) - \
                    int(infos[i].split(",")[0])
                height = int(infos[i].split(",")[3]) - \
                    int(infos[i].split(",")[1])
                dataSet.append([width, height])
        result = np.array(dataSet)
        f.close()
        return result

    def visualization(self, all_boxes_wh, centers, current_nearest):
        # 可视化数据
        figure, ax = plt.subplots(figsize=(19, 10))
        multiple = 2.5
        fontsize = 15 * multiple
        x_size = 50 * (multiple+1)
        font = {'family': 'Times New Roman',
                 'weight': 'normal',
                 'size': fontsize,}
        plt.grid(linestyle="--")
        # 设置坐标刻度值的大小以及刻度值的字体
        plt.tick_params(labelsize=fontsize)
        labels = ax.get_xticklabels() + ax.get_yticklabels()
        [label.set_fontname('Times New Roman') for label in labels]
        plt.scatter(all_boxes_wh[:, 0], all_boxes_wh[:, 1], c=current_nearest, marker='o', cmap='plasma')
        plt.scatter(centers[:, 0], centers[:, 1], color='k', marker='x', s=x_size)
        plt.title("Anchors by Kmeans", font)
        # plt.plot(C_pool, diff_c_accuracy_scores)
        # plt.xticks(C_pool, rotation=45)
        plt.xlabel('width', font)
        plt.ylabel('height', font)
        # plt.show(0)
        plt.savefig("Anchors-by-Kmeans.png")  # 保存

    def txt2clusters(self):
        if not os.path.exists(self.save_path):
            all_boxes_wh = self.txt2boxes()
            centers, current_nearest = self.kmeans(all_boxes_wh, k=self.cluster_number)
            data = (all_boxes_wh, centers, current_nearest)
            np.save(save_path, data)
        data = np.load(self.save_path)
        all_boxes_wh, centers, current_nearest = data
        self.visualization(all_boxes_wh, centers, current_nearest)
        # print(centers)
        centers = centers[np.lexsort(centers.T[0, None])]

        self.result2txt(centers)
        print("{} anchors:\n {}".format(len(centers), centers))
        print("Accuracy: {:.2f}%".format(
            self.avg_iou(all_boxes_wh, centers) * 100))


if __name__ == "__main__":
    cluster_number = 9
    filename = "train.txt"
    save_path = './kmeans.npy'
    kmeans = YOLO_Kmeans(cluster_number, filename, save_path)
    kmeans.txt2clusters()
