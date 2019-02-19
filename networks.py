import tensorflow as tf
import numpy as np
import os
from ops import *
from utils import *
import time
import ExpShapePoseNet as ESP

#-----------------------------m4_BE_GAN_network-----------------------------
#---------------------------------------------------------------------------
slim = tf.contrib.slim
class m4_BE_GAN_network:
    def __init__(self, sess, cfg):
        self.sess = sess
        self.cfg = cfg
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        self.conv_hidden_num = cfg.conv_hidden_num
        self.data_format = cfg.data_format
        self.z_dim = cfg.z_dim
        self.gamma = self.cfg.gamma
        self.lambda_k = self.cfg.lambda_k

    def build_model(self, images, labels, z):
        with tf.device('/cpu:0'):

            _, height, width, self.channel = \
                self.get_conv_shape(images, self.data_format)
            self.repeat_num = int(np.log2(height)) - 2

            self.g_lr = tf.Variable(self.cfg.g_lr, name='g_lr')
            self.d_lr = tf.Variable(self.cfg.d_lr, name='d_lr')

            self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr * 0.5, self.cfg.lr_lower_boundary),
                                         name='g_lr_update')
            self.d_lr_update = tf.assign(self.d_lr, tf.maximum(self.d_lr * 0.5, self.cfg.lr_lower_boundary),
                                         name='d_lr_update')
            self.k_t = tf.Variable(0., trainable=False, name='k_t')

            self.op_g = tf.train.AdamOptimizer(learning_rate=self.g_lr)
            self.op_d = tf.train.AdamOptimizer(learning_rate=self.d_lr)

            grads_g = []
            grads_d = []
            grads_c = []

            for i in range(self.cfg.num_gpus):
                images_on_one_gpu = images[self.cfg.batch_size * i:self.cfg.batch_size * (i + 1)]
                labels_on_one_gpu = labels[self.cfg.batch_size * i:self.cfg.batch_size * (i + 1)]
                z_on_one_gpu = z[self.cfg.batch_size * i:self.cfg.batch_size * (i + 1)]

                with tf.device("/gpu:{}".format(i)):
                    with tf.variable_scope("GPU_0") as scope:
                        if i != 0:
                            scope.reuse_variables()

                        G, self.G_var = self.GeneratorCNN(
                            z_on_one_gpu, self.conv_hidden_num, self.channel,
                            self.repeat_num, self.data_format, reuse=False)

                        if i == 0:
                            self.sampler,self.G_var = self.GeneratorCNN(z_on_one_gpu, self.conv_hidden_num, self.channel,
                                                             self.repeat_num, self.data_format, reuse=True)

                        d_out, self.D_z, self.D_var = self.DiscriminatorCNN(
                            tf.concat([G, images_on_one_gpu], 0), self.channel, self.z_dim, self.repeat_num,
                            self.conv_hidden_num, self.data_format)
                        AE_G, AE_x = tf.split(d_out, 2)

                        self.d_loss_real = tf.reduce_mean(tf.abs(AE_x - images_on_one_gpu))
                        self.d_loss_fake = tf.reduce_mean(tf.abs(AE_G - G))

                        self.d_loss = self.d_loss_real - self.k_t * self.d_loss_fake
                        self.g_loss = tf.reduce_mean(tf.abs(AE_G - G))


                        self.image_fake_sum = tf.summary.image('image_fake', AE_G)
                        self.g_loss_sum = tf.summary.scalar('g_loss', self.g_loss)
                        self.d_loss_sum = tf.summary.scalar('d_loss', self.d_loss)

                        t_vars = tf.trainable_variables()
                        self.g_vars = [var for var in t_vars if 'generator' in var.name]
                        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
                        grad_g = self.op_g.compute_gradients(self.g_loss, var_list=self.g_vars)
                        grads_g.append(grad_g)
                        grad_d = self.op_d.compute_gradients(self.d_loss, var_list=self.d_vars)
                        grads_d.append(grad_d)
                print('Init GPU:{} finshed'.format(i))
                time.sleep(1)
        mean_grad_g = m4_average_grads(grads_g)
        mean_grad_d = m4_average_grads(grads_d)
        self.g_optim = self.op_g.apply_gradients(mean_grad_g)
        self.d_optim = self.op_d.apply_gradients(mean_grad_d,global_step=self.global_step)

        self.balance = self.gamma * self.d_loss_real - self.g_loss
        self.measure = self.d_loss_real + tf.abs(self.balance)
        self.measure_sum = tf.summary.scalar('measure', self.measure)
        with tf.control_dependencies([self.d_optim, self.g_optim]):
            self.k_update = tf.assign(
                self.k_t, tf.clip_by_value(self.k_t + self.lambda_k * self.balance, 0, 1))

    def GeneratorCNN(self, z, hidden_num, output_num, repeat_num, data_format, reuse):
        with tf.variable_scope("generator", reuse=reuse) as vs:
            num_output = int(np.prod([8, 8, hidden_num]))
            x = slim.fully_connected(z, num_output, activation_fn=None)
            x = self.reshape(x, 8, 8, hidden_num, data_format)

            for idx in range(repeat_num):
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                if idx < repeat_num - 1:
                    x = self.upscale(x, 2, data_format)

            out = slim.conv2d(x, 3, 3, 1, activation_fn=None, data_format=data_format)

        variables = tf.contrib.framework.get_variables(vs)
        return out, variables

    def DiscriminatorCNN(self, x, input_channel, z_num, repeat_num, hidden_num, data_format):
        with tf.variable_scope("discriminator") as vs:
            # Encoder
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)

            prev_channel_num = hidden_num
            for idx in range(repeat_num):
                channel_num = hidden_num * (idx + 1)
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                if idx < repeat_num - 1:
                    x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format)
                    # x = tf.contrib.layers.max_pool2d(x, [2, 2], [2, 2], padding='VALID')

            x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
            z = x = slim.fully_connected(x, z_num, activation_fn=None)

            # Decoder
            num_output = int(np.prod([8, 8, hidden_num]))
            x = slim.fully_connected(x, num_output, activation_fn=None)
            x = self.reshape(x, 8, 8, hidden_num, data_format)

            for idx in range(repeat_num):
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                if idx < repeat_num - 1:
                    x = self.upscale(x, 2, data_format)

            out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)

        variables = tf.contrib.framework.get_variables(vs)
        return out, z, variables

    def get_conv_shape(self,tensor, data_format):
        shape = self.int_shape(tensor)
        # always return [N, H, W, C]
        if data_format == 'NCHW':
            return [shape[0], shape[2], shape[3], shape[1]]
        elif data_format == 'NHWC':
            return shape

    def upscale(self,x, scale, data_format):
        _, h, w, _ = self.get_conv_shape(x, data_format)
        return self.resize_nearest_neighbor(x, (h * scale, w * scale), data_format)

    def int_shape(self,tensor):
        shape = tensor.get_shape().as_list()
        return [num if num is not None else -1 for num in shape]

    def reshape(self,x, h, w, c, data_format):
        if data_format == 'NCHW':
            x = tf.reshape(x, [-1, c, h, w])
        else:
            x = tf.reshape(x, [-1, h, w, c])
        return x

    def resize_nearest_neighbor(self, x, new_size, data_format):
        if data_format == 'NCHW':
            x = nchw_to_nhwc(x)
            x = tf.image.resize_nearest_neighbor(x, new_size)
            x = nhwc_to_nchw(x)
        else:
            x = tf.image.resize_nearest_neighbor(x, new_size)
        return x