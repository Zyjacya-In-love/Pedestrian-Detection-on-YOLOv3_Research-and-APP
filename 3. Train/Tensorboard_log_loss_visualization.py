import os
from tensorboard.backend.event_processing import event_accumulator
import numpy as np
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser(description='Extract Tensorboard log and draw the loss curve.')
parser.add_argument('log_pre_50', help='Path to log of the first 50 epochs.')
parser.add_argument('log_after', help='Path to the rest of log after 50 epochs.')



def extract_loss(log_name):
    # 加载日志数据
    ea = event_accumulator.EventAccumulator(log_name)
    ea.Reload()
    # print(ea.scalars.Keys())

    # val_loss = ea.scalars.Items('val_loss')
    # print(len(val_loss))
    # print([(i.step, i.value) for i in val_loss])
    val_loss = ea.scalars.Items('val_loss')
    loss=ea.scalars.Items('loss')
    return [i.value for i in val_loss], [i.value for i in loss]

def loss_curve(val_loss, loss, result_path):
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    fig, ax = plt.subplots(figsize=(16, 10))
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

    x = [i+1 for i in range(len(loss))]
    x_lab = [i for i in range(0, len(loss)+10, 10)]
    plt.xticks(x_lab)
    ax.plot(x, val_loss, label='val_loss')
    ax.plot(x, loss, label='loss')
    # ax.plot(result['loss'].values, label='loss')
    # plt.yticks(y_ticks)  # 如果不想自己设置纵坐标，可以注释掉。
    plt.grid(linestyle="--")
    ax.legend(prop = font, loc='best')
    ax.set_title('The loss curves', font)
    ax.set_xlabel('Epoch', font)
    fig.savefig(result_path + 'loss.png')




def _main(args):
    # log_name_50 = r'events.out.tfevents.1586104689.r1cmpsrvs79-14ig0702'
    # log_name_109 = r'events.out.tfevents.1586671921.r1cmpsrvs79-14ig0702'
    log_name_50 = args.log_pre_50
    log_name_109 = args.log_after
    val_loss50, loss50 = extract_loss(log_name_50)
    val_loss109, loss109 = extract_loss(log_name_109)
    val_loss = val_loss50 + val_loss109
    loss = loss50 + loss109
    result_path = './'
    loss_curve(val_loss, loss, result_path)



if __name__ == '__main__':
    _main(parser.parse_args())