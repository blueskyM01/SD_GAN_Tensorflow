import os
import json
import tensorflow as tf
import numpy as np
from collections import defaultdict
import time
import cv2

class tfrecords_maker:
    def __init__(self, data_dir, dataset_dir, dataset_name, label_dir, label_name):
        """
        Introduction
        ------------
            构造函数
        Parameters
        ----------
            data_dir: 文件路径
            mode: 数据集模式 "train"
            anchors: 数据集聚类得到的anchor
            num_classes: 数据集图片类别数量
            input_shape: 图像输入模型的大小
            max_boxes: 每张图片最大的box数量
            jitter: 随机长宽比系数
            hue: 调整hsv颜色空间系数
            sat: 调整饱和度系数
            cont: 调整对比度系数
            bri: 调整亮度系数
        """
        self.data_dir = data_dir    # model_data
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.label_dir = label_dir
        self.label_name = label_name

    def convert_to_tfrecord(self, tfrecord_path, num_tfrecords):
        """
        Introduction
        ------------
            将图片和boxes数据存储为tfRecord
        Parameters
        ----------
            tfrecord_path: tfrecord文件存储路径
            num_tfrecords: 分成多少个tfrecord
        """
        # image_data, boxes_data = self.read_annotations()
        # images_num = int(len(image_data) / num_tfrecords)

        names = np.loadtxt(os.path.join(self.label_dir, self.label_name), dtype=np.str)
        num_image = names.shape[0]
        images_num = int(num_image / num_tfrecords)
        image_data, labels = self.m4_get_file_label_name(self.label_dir, self.label_name, self.dataset_dir, self.dataset_name)

        for index_records in range(num_tfrecords):
            output_file = os.path.join(tfrecord_path, str(index_records) + '_' + 'faceimage' + '.tfrecords')
            with tf.python_io.TFRecordWriter(output_file) as record_writer:
                for index in range(index_records * images_num, (index_records + 1) * images_num):
                    with tf.gfile.FastGFile(image_data[index], 'rb') as file:
                        image = file.read()
                        example = tf.train.Example(features = tf.train.Features(
                            feature = {
                                'image/encoded' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
                                'image/label' : tf.train.Feature(float_list = tf.train.FloatList(value = [labels[index]])),
                            }
                        ))
                        record_writer.write(example.SerializeToString())
                        if index % 100 == 0:
                            print('Processed {} of {} images'.format(index + 1, num_image))
                print(output_file + ' is ok....')

    def m4_get_file_label_name(self, label_dir, label_name, dataset_dir, dataset_name):
        '''
        :param label_dir: label dir
        :param label_name: label name
        :param dataset_dir: dataset dir
        :param dataset_name: dataset name
        :return:filename_list, label_list
        '''
        filepath_name = os.path.join(label_dir, label_name)
        save_data_path_name = os.path.join(dataset_dir, dataset_name)
        data = np.loadtxt(filepath_name, dtype=str)
        filename = data[:, 0].tolist()
        label = data[:, 1].tolist()
        filename_list = []
        label_list = []
        for i in range(data.shape[0]):
            filename_list.append(os.path.join(save_data_path_name, filename[i].lstrip("b'").rstrip("'")))
            label_list.append(int(label[i].lstrip("b'").rstrip("'")))
        return filename_list, label_list




class Reader:
    def __init__(self, tfrecords_dir,label_dir, label_name):
        """
        Introduction
        ------------
            构造函数
        Parameters
        ----------
            data_dir: 文件路径
            mode: 数据集模式 "train"
            anchors: 数据集聚类得到的anchor
            num_classes: 数据集图片类别数量
            input_shape: 图像输入模型的大小
            max_boxes: 每张图片最大的box数量
            jitter: 随机长宽比系数
            hue: 调整hsv颜色空间系数
            sat: 调整饱和度系数
            cont: 调整对比度系数
            bri: 调整亮度系数
        """
        self.label_dir = label_dir
        self.label_name = label_name
        self.tfrecords_dir = tfrecords_dir    # model_data
        file_pattern = self.tfrecords_dir + "/*" + 'faceimage' + '.tfrecords'
        self.TfrecordFile = tf.gfile.Glob(file_pattern)


    def parser(self, serialized_example):
        """
        Introduction
        ------------
            解析tfRecord数据
        Parameters
        ----------
            serialized_example: 序列化的每条数据
        """
        features = tf.parse_single_example(
            serialized_example,
            features = {
                'image/encoded' : tf.FixedLenFeature([], dtype = tf.string),
                'image/label' : tf.VarLenFeature(dtype = tf.float32)
            }
        )
        image = tf.image.decode_jpeg(features['image/encoded'], channels = 3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32) * 2.0 - 1.0
        label = features['image/label'].values
        image = tf.image.resize_images(image, [128, 128])
        return image, label




    def build_dataset(self, batch_size, epoch, is_train=True):
        """
        Introduction
        ------------
            建立数据集dataset
        Parameters
        ----------
            batch_size: batch大小
        Return
        ------
            dataset: 返回tensorflow的dataset
        """
        names = np.loadtxt(os.path.join(self.label_dir, self.label_name), dtype=np.str)
        dataset_size = names.shape[0]

        dataset = tf.data.TFRecordDataset(filenames = self.TfrecordFile)
        dataset = dataset.map(self.parser, num_parallel_calls = 10)
        if is_train:
            dataset = dataset.shuffle(10000).batch(batch_size).repeat(epoch)
        else:
            dataset = dataset.batch(batch_size).repeat(epoch)
        # dataset = dataset.repeat().batch(batch_size).prefetch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        one_element = iterator.get_next()
        return one_element, dataset_size

# dataset_dir = '/media/yang/F/DataSet/Face'
# dataset_name = 'ms1s_align'
# label_dir = '/media/yang/F/DataSet/Face/Label'
# label_name = 'MS-Celeb-1M_clean_list.txt'
# tfrecord_path = '/media/yang/F/DataSet/Face/ms1s_tfrecords'
# tensor_file_maker = tfrecords_maker(tfrecord_path, dataset_dir, dataset_name, label_dir, label_name)
# tensor_file_maker.convert_to_tfrecord(tfrecord_path, 100)