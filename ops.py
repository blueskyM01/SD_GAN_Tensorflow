import tensorflow as tf
import os
import numpy as np


def m4_leak_relu(x, leak=0.2):
    return tf.maximum(x, leak * x)


def m4_batch_norm(input_, is_trainable):
    try:
        output = tf.contrib.layers.batch_norm(input_, decay=0.9,
                                              updates_collections=None,
                                              epsilon=1e-5,
                                              scale=True,
                                              is_training=is_trainable)
    except:
        mean, variance = tf.nn.moments(input_, axes=[0, 1, 2])
        _, _, _, nc = input_.get_shape().as_list()
        beta = tf.get_variable('beta', [nc], tf.float32,
                               initializer=tf.constant_initializer(0.0, tf.float32))  # [out_channels]
        gamma = tf.get_variable('gamma', [nc], tf.float32,
                                initializer=tf.constant_initializer(1.0, tf.float32))
        output = tf.nn.batch_normalization(input_, mean, variance, beta, gamma, 1e-5)
    return output


def m4_active_function(input_, active_function='relu', name='active_function'):
    with tf.variable_scope(name):
        if active_function == 'relu':
            active = tf.nn.relu(input_)
            return active
        elif active_function == 'leak_relu':
            active = m4_leak_relu(input_)
            return active
        elif active_function == 'elu':
            active = tf.nn.elu(input_)
            return active
        else:
            return input_


def m4_conv_moudel(input_, fiters, k_h=3, k_w=3, s_h=1, s_w=1, stddev=0.02, padding="SAME", active_function='relu',
                   norm='batch_norm', is_trainable=True, do_active=True, name='m4_conv_moudel'):
    with tf.variable_scope(name):
        conv = m4_conv(input_, fiters, k_h, k_w, s_h, s_w, padding, stddev)
        if do_active:
            conv = m4_active_function(conv, active_function)
        if norm == 'batch_norm':
            conv = m4_batch_norm(conv, is_trainable)

        return conv


def m4_conv(input_, fiters, k_h=3, k_w=3, s_h=1, s_w=1, padding="SAME", stddev=0.02, name='conv'):
    with tf.variable_scope(name) as scope:
        batch, heigt, width, nc = input_.get_shape().as_list()
        w = tf.get_variable('w', [k_h, k_w, nc, fiters], initializer=tf.truncated_normal_initializer(stddev=stddev))
        bias = tf.get_variable('bias', [fiters], initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(input_, w, strides=[1, s_h, s_w, 1], padding=padding) + bias
        vars = tf.contrib.framework.get_variables(scope)
        return conv, vars


def m4_deconv_moudel(input_, output_shape, k_h=3, k_w=3, s_h=2, s_w=2, padding="SAME", stddev=0.02,
                     active_function='relu',
                     norm='batch_norm', is_trainable=True, do_active=True, name='m4_deconv_moudel'):
    with tf.variable_scope(name):
        deconv = m4_deconv(input_, output_shape, k_h, k_w, s_h, s_w, padding, stddev)
        if do_active:
            deconv = m4_active_function(deconv, active_function)
        if norm == 'batch_norm':
            deconv = m4_batch_norm(deconv, is_trainable)
        return deconv


def m4_deconv(input_, output_shape, k_h=3, k_w=3, s_h=2, s_w=2, padding="SAME", stddev=0.02, name='m4_deconv'):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))
        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                            strides=[1, s_h, s_w, 1])

            # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                    strides=[1, s_h, s_w, 1])
        deconv = deconv + biases
        return deconv


# def m4_linear(input_, output, active_function='leak_relu', norm='batch_norm', is_trainable=True, do_active=True,
#               stddev=0.02, name='m4_linear'):
#     with tf.variable_scope(name):
#         input_shape = input_.get_shape().as_list()
#         w = tf.get_variable('w', [input_shape[-1], output], initializer=tf.random_normal_initializer(stddev=stddev))
#         biases = tf.get_variable('biases', [output], initializer=tf.constant_initializer(0.0))
#         conn = tf.matmul(input_, w) + biases
#         if do_active:
#             conn = m4_active_function(conn, active_function)
#         if norm == 'batch_norm':
#             conn = m4_batch_norm(conn, is_trainable)

def m4_FullConnect(input_, output, active_function='leak_relu', stddev=0.02, reuse=False, name='fullconnect'):
    with tf.variable_scope(name, reuse=reuse):
        input_shape = input_.get_shape().as_list()
        w = tf.get_variable('w', [input_shape[-1], output], initializer=tf.random_normal_initializer(stddev=stddev))
        biases = tf.get_variable('biases', [output], initializer=tf.constant_initializer(0.0))
        conn = tf.matmul(input_, w) + biases
        conn = m4_active_function(conn, active_function)
        return conn


def m4_resnet_18(input_, name='resnet_18'):
    with tf.variable_scope(name):
        conv1 = m4_conv_moudel(input_, 32, k_h=3, k_w=3, s_h=1, s_w=1, stddev=0.02, padding="SAME",
                               active_function='leak_relu',
                               norm='batch_norm', is_trainable=True, do_active=True, name='conv1')
        pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='pool1')  # 125x125x32
        conv2 = m4_conv_moudel(pool1, 64, k_h=3, k_w=3, s_h=1, s_w=1, stddev=0.02, padding="SAME",
                               active_function='leak_relu',
                               norm='batch_norm', is_trainable=True, do_active=True, name='conv2')
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='pool2')  # 63x63x64
        res_block1 = pool2

        for i in range(2):
            res_block1 = m4_res_block(res_block1, [64, 64], [3, 3], [1, 1], active_function='leak_relu',
                                      name='3x3x64_{}'.format(i))

        conv3 = m4_conv_moudel(res_block1, 128, k_h=3, k_w=3, s_h=2, s_w=2, stddev=0.02, padding="SAME",
                               active_function='leak_relu',
                               norm='batch_norm', is_trainable=True, do_active=True, name='conv3')  ## 32x32x128

        res_block2 = conv3
        for i in range(2):
            res_block2 = m4_res_block(res_block2, [128, 128], [3, 3], [1, 1], active_function='leak_relu',
                                      name='3x3x128_{}'.format(i))
        conv4 = m4_conv_moudel(res_block2, 256, k_h=3, k_w=3, s_h=2, s_w=2, stddev=0.02, padding="SAME",
                               active_function='leak_relu',
                               norm='batch_norm', is_trainable=True, do_active=True, name='conv4')  # 16x16x256
        res_block3 = conv4
        for i in range(2):
            res_block3 = m4_res_block(res_block3, [256, 256], [3, 3], [1, 1], active_function='leak_relu',
                                      name='3x3x256_{}'.format(i))
        conv5 = m4_conv_moudel(res_block3, 512, k_h=3, k_w=3, s_h=2, s_w=2, stddev=0.02, padding="SAME",
                               active_function='leak_relu',
                               norm='batch_norm', is_trainable=True, do_active=True, name='conv5')  # 8x8x512
        res_block4 = conv5
        for i in range(3):
            res_block4 = m4_res_block(res_block4, [512, 512], [3, 3], [1, 1], active_function='leak_relu',
                                      name='3x3x512_{}'.format(i))
        _, h, w, nc = res_block4.get_shape().as_list()
        reshape = tf.reshape(res_block4, [-1, h * w * nc])
        conn1 = m4_linear(reshape, 256, name='conn1')
        output = m4_linear(conn1, 1, do_active=False, norm=None, name='output')

        return conn1, output


def m4_res_block(input_, n_filters, k_sizes, s_sizes, padding='SAME', stddev=0.02, active_function='relu',
                 norm='batch_norm',
                 is_trainable=True, do_active=True, name='m4_res_block'):
    # n_filters=[64,64]
    # k_sizes=[3,3]
    # s_sizes=[1,1]
    with tf.variable_scope(name):
        conv = input_
        for i, (nf, k_size, s_size) in enumerate(zip(n_filters, k_sizes, s_sizes)):
            if i < (len(n_filters) - 1):
                do_active = True
                norm = 'batch_norm'
            else:
                do_active = False
                norm = None
            conv = m4_conv_moudel(conv, nf, k_h=k_size, k_w=k_size, s_h=s_size, s_w=s_size, stddev=stddev,
                                  padding=padding,
                                  active_function=active_function, norm=norm, is_trainable=is_trainable,
                                  do_active=do_active,
                                  name='{}'.format(i))

        conv = tf.nn.relu(conv + input_, name='relu')
        conv = m4_batch_norm(conv, is_trainable)
        return conv


def m4_VGG(input_, name='VGG'):
    with tf.variable_scope(name):
        conv1 = m4_conv_moudel(input_, 32, k_h=3, k_w=3, s_h=1, s_w=1, stddev=0.02, padding="SAME",
                               active_function='leak_relu',
                               norm='batch_norm', is_trainable=True, do_active=True, name='conv1')
        pool1 = tf.nn.max_pool(conv1, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='pool1')  # 125x125x32
        conv2 = m4_conv_moudel(pool1, 64, k_h=3, k_w=3, s_h=1, s_w=1, stddev=0.02, padding="SAME",
                               active_function='leak_relu',
                               norm='batch_norm', is_trainable=True, do_active=True, name='conv2')
        pool2 = tf.nn.max_pool(conv2, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='pool2')  # 63x63x64

        conv3 = m4_conv_moudel(pool2, 64, k_h=3, k_w=3, s_h=1, s_w=1, stddev=0.02, padding="SAME",
                               active_function='leak_relu',
                               norm='batch_norm', is_trainable=True, do_active=True, name='conv3')
        pool3 = tf.nn.max_pool(conv3, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME', name='pool2')  # 32x32x64

        conv4 = m4_conv_moudel(pool3, 128, k_h=3, k_w=3, s_h=1, s_w=1, stddev=0.02, padding="SAME",
                               active_function='leak_relu',
                               norm='batch_norm', is_trainable=True, do_active=True, name='conv4')  # 32x32x128
        conv5 = m4_conv_moudel(conv4, 128, k_h=3, k_w=3, s_h=1, s_w=1, stddev=0.02, padding="SAME",
                               active_function='leak_relu',
                               norm='batch_norm', is_trainable=True, do_active=True, name='conv5')  # 32x32x128

        conv6 = m4_conv_moudel(conv5, 128, k_h=3, k_w=3, s_h=2, s_w=2, stddev=0.02, padding="SAME",
                               active_function='leak_relu',
                               norm='batch_norm', is_trainable=True, do_active=True, name='conv6')  # 16x16x128
        return conv5, tf.reduce_mean(conv6)


def m4_average_grads(tower):
    averaged_grads = []
    for grads_and_vars in zip(*tower):
        # print(grads_and_vars)
        grads = []
        for g, _ in grads_and_vars:
            expanded_grad = tf.expand_dims(g, 0, 'expand_grads')
            grads.append(expanded_grad)
        grad = tf.concat(values=grads, axis=0)
        grad = tf.reduce_mean(input_tensor=grad, axis=0, keep_dims=False)
        g_and_v = (grad, grads_and_vars[0][1])
        averaged_grads.append(g_and_v)
    return averaged_grads


def m4_wgan_loss(d_real, d_fake):
    # Standard WGAN loss
    g_loss = -tf.reduce_mean(d_fake)
    d_loss = tf.reduce_mean(d_fake) - tf.reduce_mean(d_real)
    return d_loss, g_loss


def m4_parse_function(filename, label):
    image_string = tf.read_file(filename)
    # image_decoded = tf.image.decode_image(image_string)
    image_decoded = tf.image.decode_jpeg(image_string, 3)
    image_decoded = tf.image.convert_image_dtype(image_decoded, dtype=tf.float32) * 2.0 - 1.0
    image_resized = tf.image.resize_images(image_decoded, [128, 128])
    label = tf.one_hot(label, 10575)
    return image_resized, label


def m4_feat_norm(input_):
    # value range:0~1
    # a / (a^2 + b^2)
    x = input_ / tf.sqrt((tf.reduce_sum(tf.square(input_))))
    return x
