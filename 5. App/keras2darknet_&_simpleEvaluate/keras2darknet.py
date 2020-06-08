# -*- coding: utf-8 -*- 
import argparse
import os

import numpy as np
import keras
from keras.models import load_model

import keras.backend as K




parser = argparse.ArgumentParser(description="Keras model *.h5 + *.cfg convert to Darknet *.weights!!!")
parser.add_argument('-cfg', dest='cfg_path', default='model/yolo-person.cfg', help="Path to Darknet cfg")
parser.add_argument('-mh5', dest='model_h5_path', default='model/trained_model_final.h5', help="Path to keras model *.h5")
parser.add_argument('-sf', dest='save_file', default='model/yolov3-keras2darknet.weights', help='final save *.weights file')




class WeightSaver(object):

    def __init__(self, model_h5_path, save_file):

        self.model = load_model(model_h5_path, compile=False)

        self.layers = {weight.name:weight for weight in self.model.weights}
        self.sess = K.get_session()
        self.fhandle = open(save_file,'wb')
        self._write_head()

    def _write_head(self):
        numpy_data = np.ndarray(shape=(3,),
                          dtype='int32',
                          buffer=np.array([0,2,0],dtype='int32') )
        self.save(numpy_data)
        numpy_data = np.ndarray(shape=(1,),
                          dtype='int64',
                          buffer=np.array([320000],dtype='int64'))
        self.save(numpy_data)

    def get_bn_layername(self,num):
        layer_name = 'batch_normalization_{num}'.format(num=num)
        bias = self.layers['{0}/beta:0'.format(layer_name)]
        scale = self.layers['{0}/gamma:0'.format(layer_name)]
        mean = self.layers['{0}/moving_mean:0'.format(layer_name)]
        var = self.layers['{0}/moving_variance:0'.format(layer_name)]
       
        bias_np = self.get_numpy(bias)
        scale_np = self.get_numpy(scale)
        mean_np = self.get_numpy(mean)
        var_np = self.get_numpy(var)
        return bias_np,scale_np,mean_np,var_np

    def get_convbias_layername(self,num):
        layer_name = 'conv2d_{num}'.format(num=num)
        bias = self.layers['{0}/bias:0'.format(layer_name)]
      
        bias_np = self.get_numpy(bias)
        return bias_np

    def get_conv_layername(self,num):
        layer_name = 'conv2d_{num}'.format(num=num)
        conv = self.layers['{0}/kernel:0'.format(layer_name)]
       
        conv_np = self.get_numpy(conv)
        return conv_np

    def get_numpy(self,layer_name):
        numpy_data = self.sess.run(layer_name)
        return numpy_data

    def save(self,numpy_data):
        bytes_data = numpy_data.tobytes()
        self.fhandle.write(bytes_data)
        self.fhandle.flush()

    def close(self):
        self.fhandle.close()


class KerasParser(object):

    def __init__(self, cfg_path, model_h5_path, save_file):
        self.block_gen = self._get_block(cfg_path)
        self.weights_saver = WeightSaver(model_h5_path, save_file)
        self.count_conv = 0
        self.count_bn = 0

    def _get_block(self,cfg_path):

        block = {}
        with open(cfg_path,'r', encoding='utf-8') as fr:
            for line in fr:
                line = line.strip()
                if '[' in line and ']' in line:
                    if block:
                        yield block
                    block = {}
                    block['type'] = line.strip(' []')
                elif not line or '#' in line:
                    continue
                else:
                    key,val = line.strip().replace(' ','').split('=')
                    key,val = key.strip(), val.strip()
                    block[key] = val

            yield block

    def close(self):
        self.weights_saver.close()

    def conv(self, block):
        self.count_conv += 1
        batch_normalize = 'batch_normalize' in block
        # print('handing.. ',self.count_conv)

        # 如果bn存在，则先处理bn，顺序为bias，scale，mean，var
        if batch_normalize:
            bias,scale,mean,var = self.bn()
            self.weights_saver.save(bias)
            
            scale = scale.reshape(1,-1)
            mean = mean.reshape(1,-1)
            var = var.reshape(1,-1)
            remain = np.concatenate([scale,mean,var],axis=0)
            self.weights_saver.save(remain)

        # 否则，先处理biase
        else:
            conv_bias = self.weights_saver.get_convbias_layername(self.count_conv)
            self.weights_saver.save(conv_bias)

        # 接着处理weights
        conv_weights = self.weights_saver.get_conv_layername(self.count_conv)
        # 需要将(height, width, in_dim, out_dim)转换成(out_dim, in_dim, height, width)
        conv_weights = np.transpose(conv_weights,[3,2,0,1])
        self.weights_saver.save(conv_weights)

    def bn(self):
        self.count_bn += 1
        bias,scale,mean,var = self.weights_saver.get_bn_layername(self.count_bn) 
        return bias,scale,mean,var



def _main(args):
    keras_loader = KerasParser(args.cfg_path, args.model_h5_path, args.save_file)

    for block in keras_loader.block_gen:
        if 'convolutional' in block['type']:
            keras_loader.conv(block)
    keras_loader.close()



if __name__ == '__main__':
    _main(parser.parse_args())