import tensorflow as tf
import numpy as np
import os

def data_loader(label_dir,label_name,dataset_dir,dataset_name,batch_size=16,epoch=100,buffer_size=10000):
    '''

    :param label_dir:
    :param label_name:
    :param dataset_dir:
    :param dataset_name:
    :param buffer_size:
    :param batch_size:
    :param epoch:
    :return:
    '''
    names = np.loadtxt(os.path.join(label_dir, label_name), dtype=np.str)
    dataset_size = names.shape[0]
    names, labels = m4_get_file_label_name(label_dir, label_name, dataset_dir, dataset_name)
    filenames = tf.constant(names)
    filelabels = tf.constant(labels)
    try:
        dataset = tf.data.Dataset.from_tensor_slices((filenames, filelabels))
    except:
        dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, filelabels))

    dataset = dataset.map(m4_parse_function)
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size).repeat(epoch)
    iterator = dataset.make_one_shot_iterator()
    one_element = iterator.get_next()
    return one_element,dataset_size

def m4_get_file_label_name(label_dir,label_name,dataset_dir,dataset_name):
    '''
    :param label_dir: label dir
    :param label_name: label name
    :param dataset_dir: dataset dir
    :param dataset_name: dataset name
    :return:filename_list, label_list
    '''
    filepath_name = os.path.join(label_dir,label_name)
    save_data_path_name = os.path.join(dataset_dir,dataset_name)
    data = np.loadtxt(filepath_name,dtype=str)
    filename = data[:,0].tolist()
    label=data[:,1].tolist()
    filename_list = []
    label_list=[]
    for i in range(data.shape[0]):
        filename_list.append(os.path.join(save_data_path_name,filename[i].lstrip("b'").rstrip("'")))
        label_list.append(int(label[i].lstrip("b'").rstrip("'")))
    return filename_list,label_list

def m4_parse_function(filename, label):
    image_string = tf.read_file(filename)
    # image_decoded = tf.image.decode_image(image_string)
    image_decoded = tf.image.decode_jpeg(image_string,3)
    image_decoded = tf.image.convert_image_dtype(image_decoded,dtype=tf.float32) * 2.0 - 1.0
    image_resized = tf.image.resize_images(image_decoded, [128, 128])
    label = tf.one_hot(label, 10575)
    return image_resized, label