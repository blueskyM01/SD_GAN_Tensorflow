import tensorflow as tf
import numpy as np
import os
from ops import *
from utils import *
import time
import ExpShapePoseNet as ESP
import importlib
from collections import defaultdict

# -----------------------------m4_BE_GAN_network-----------------------------
# ---------------------------------------------------------------------------
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
        self.resolution_ = 'resolution_'
        self.g_lr = tf.Variable(self.cfg.g_lr, name='g_lr')
        self.d_lr = tf.Variable(self.cfg.d_lr, name='d_lr')
        self.expr_shape_pose = ESP.m4_3DMM(self.cfg)

    def build_model(self, images, labels, z):
        muti_gpu_reuse_0 = False
        muti_gpu_reuse_1 = True
        _, height, width, self.channel = \
            self.get_conv_shape(images, self.data_format)
        self.repeat_num = int(np.log2(height)) - 2

        self.g_lr_update = tf.assign(self.g_lr, tf.maximum(self.g_lr * 0.8, self.cfg.lr_lower_boundary),
                                     name='g_lr_update')
        self.d_lr_update = tf.assign(self.d_lr, tf.maximum(self.d_lr * 0.8, self.cfg.lr_lower_boundary),
                                     name='d_lr_update')
        self.k_t_128 = tf.Variable(0., trainable=False, name='k_t_128')
        self.k_t_64 = tf.Variable(0., trainable=False, name='k_t_64')
        self.k_t_32 = tf.Variable(0., trainable=False, name='k_t_32')
        self.k_t_16 = tf.Variable(0., trainable=False, name='k_t_16')
        self.k_t_8 = tf.Variable(0., trainable=False, name='k_t_8')
        self.op_g = tf.train.AdamOptimizer(learning_rate=self.g_lr)
        self.op_d = tf.train.AdamOptimizer(learning_rate=self.d_lr)

        grads_g_128 = []
        grads_d_128 = []

        grads_g_64 = []
        grads_d_64 = []

        grads_g_32 = []
        grads_d_32 = []

        grads_g_16 = []
        grads_d_16 = []

        grads_g_8 = []
        grads_d_8 = []

        for i in range(self.cfg.num_gpus):
            images_on_one_gpu = images[self.cfg.batch_size * i:self.cfg.batch_size * (i + 1)]
            z_on_one_gpu = z[self.cfg.batch_size * i:self.cfg.batch_size * (i + 1)]

            with tf.device("/gpu:{}".format(i)):
                if i == 0:
                    muti_gpu_reuse_0 = False
                    muti_gpu_reuse_1 = True
                else:
                    muti_gpu_reuse_0 = True
                    muti_gpu_reuse_1 = True

                # 获取各个分辨率图像：image_real
                img_8, img_16, img_32, img_64, img_128 = self.m4_various_resolution_image(images_on_one_gpu)

                # 提取各分辨率ID的特征
                id_feat_real = self.m4_ID_Extractor(img_128, reuse=muti_gpu_reuse_0)
                # 3DMM 特征提取
                shape_real_norm, expr_real_norm, pose_real_norm = self.model_3DMM_default_graph(self.cfg.batch_size,
                                                                                                self.expr_shape_pose,
                                                                                                img_128, reuse=muti_gpu_reuse_0)


                z_concat_feat = tf.concat([z_on_one_gpu, shape_real_norm, pose_real_norm, expr_real_norm, id_feat_real],
                                          axis=1)
                self.G = self.GeneratorCNN(z_concat_feat, self.conv_hidden_num, self.channel, self.repeat_num,
                                           self.data_format, reuse=muti_gpu_reuse_0, name_='generator')

                img_fake_zoom_128 = self.resize_nearest_neighbor(self.G[self.resolution_ + '128'][0], (128, 128),
                                                                    self.data_format)
                img_fake_zoom_64 = self.resize_nearest_neighbor(self.G[self.resolution_ + '64'][0], (128,128),
                                                                    self.data_format)
                img_fake_zoom_32 = self.resize_nearest_neighbor(self.G[self.resolution_ + '32'][0], (128, 128),
                                                                    self.data_format)
                img_fake_zoom_16 = self.resize_nearest_neighbor(self.G[self.resolution_ + '16'][0], (128, 128),
                                                                    self.data_format)
                img_fake_zoom_8 = self.resize_nearest_neighbor(self.G[self.resolution_ + '8'][0], (128, 128),
                                                                    self.data_format)

                G_all = tf.concat([img_fake_zoom_128, img_fake_zoom_64, img_fake_zoom_32, img_fake_zoom_16, img_fake_zoom_8],
                                  axis=0)

                id_feat_fake = self.m4_ID_Extractor(G_all, reuse=muti_gpu_reuse_1)
                id_feat_fake_128, id_feat_fake_64, id_feat_fake_32, id_feat_fake_16, id_feat_fake_8 = tf.split(id_feat_fake, 5)

                shape_fake_norm, expr_fake_norm, pose_fake_norm = self.model_3DMM_default_graph(self.cfg.batch_size * 5,
                    self.expr_shape_pose, G_all, reuse=muti_gpu_reuse_1)

                shape_fake_norm_128, shape_fake_norm_64, shape_fake_norm_32, shape_fake_norm_16, shape_fake_norm_8 = \
                    tf.split(shape_fake_norm, 5)
                expr_fake_norm_128, expr_fake_norm_64, expr_fake_norm_32, expr_fake_norm_16, expr_fake_norm_8 = \
                    tf.split(expr_fake_norm, 5)
                pose_fake_norm_128, pose_fake_norm_64, pose_fake_norm_32, pose_fake_norm_16, pose_fake_norm_8 = \
                    tf.split(pose_fake_norm, 5)

                g_fake_128 = self.G[self.resolution_ + '128'][0]
                g_fake_64 = self.G[self.resolution_ + '64'][0]
                g_fake_32 = self.G[self.resolution_ + '32'][0]
                g_fake_16 = self.G[self.resolution_ + '16'][0]
                g_fake_8 = self.G[self.resolution_ + '8'][0]
                # 拼接变量：image_real and image_fake
                self.G[self.resolution_ + '128'][0] = tf.concat([self.G[self.resolution_ + '128'][0], img_128], 0)
                # self.G[self.resolution_ + '64'][0] = tf.concat([self.G[self.resolution_ + '64'][0], img_64], 0)
                # self.G[self.resolution_ + '32'][0] = tf.concat([self.G[self.resolution_ + '32'][0], img_32], 0)
                # self.G[self.resolution_ + '16'][0] = tf.concat([self.G[self.resolution_ + '16'][0], img_16], 0)
                # self.G[self.resolution_ + '8'][0] = tf.concat([self.G[self.resolution_ + '8'][0], img_8], 0)


                if i == 0:
                    self.sampler = g_fake_128

                self.shape_loss_128 = tf.reduce_mean(tf.square(shape_real_norm - shape_fake_norm_128))
                self.expr_loss_128 = tf.reduce_mean(tf.square(expr_real_norm - expr_fake_norm_128))
                self.pose_loss_128 = tf.reduce_mean(tf.square(pose_real_norm - pose_fake_norm_128))
                self.id_loss_128 = tf.reduce_mean(tf.square(id_feat_real - id_feat_fake_128))

                self.shape_loss_64 = tf.reduce_mean(tf.square(shape_real_norm - shape_fake_norm_64))
                self.expr_loss_64 = tf.reduce_mean(tf.square(expr_real_norm - expr_fake_norm_64))
                self.pose_loss_64 = tf.reduce_mean(tf.square(pose_real_norm - pose_fake_norm_64))
                self.id_loss_64 = tf.reduce_mean(tf.square(id_feat_real - id_feat_fake_64))

                self.shape_loss_32 = tf.reduce_mean(tf.square(shape_real_norm - shape_fake_norm_32))
                self.expr_loss_32 = tf.reduce_mean(tf.square(expr_real_norm - expr_fake_norm_32))
                self.pose_loss_32 = tf.reduce_mean(tf.square(pose_real_norm - pose_fake_norm_32))
                self.id_loss_32 = tf.reduce_mean(tf.square(id_feat_real - id_feat_fake_32))

                self.shape_loss_16 = tf.reduce_mean(tf.square(shape_real_norm - shape_fake_norm_16))
                self.expr_loss_16 = tf.reduce_mean(tf.square(expr_real_norm - expr_fake_norm_16))
                self.pose_loss_16 = tf.reduce_mean(tf.square(pose_real_norm - pose_fake_norm_16))
                self.id_loss_16 = tf.reduce_mean(tf.square(id_feat_real - id_feat_fake_16))

                self.shape_loss_8 = tf.reduce_mean(tf.square(shape_real_norm - shape_fake_norm_8))
                self.expr_loss_8 = tf.reduce_mean(tf.square(expr_real_norm - expr_fake_norm_8))
                self.pose_loss_8 = tf.reduce_mean(tf.square(pose_real_norm - pose_fake_norm_8))
                self.id_loss_8 = tf.reduce_mean(tf.square(id_feat_real - id_feat_fake_8))

                d_out = self.DiscriminatorCNN(
                    self.G, self.channel, self.z_dim, self.repeat_num,
                    self.conv_hidden_num, self.data_format, reuse=muti_gpu_reuse_0, name_='discriminator')

                AE_G_8, AE_x_8 = tf.split(d_out[self.resolution_ + '8'][0], 2)
                self.d_loss_real_8 = tf.reduce_mean(tf.abs(AE_x_8 - img_8))
                self.d_loss_fake_8 = tf.reduce_mean(tf.abs(AE_G_8 - g_fake_8))
                self.d_loss_8 = self.d_loss_real_8 - self.k_t_8 * self.d_loss_fake_8
                self.g_loss_8 = tf.reduce_mean(
                    tf.abs(AE_G_8 - g_fake_8)) + self.cfg.lambda_s * self.shape_loss_8 + \
                                 self.cfg.lambda_e * self.expr_loss_8 \
                                 + self.cfg.lambda_p * self.pose_loss_8 + self.cfg.lambda_id * self.id_loss_8

                AE_G_16, AE_x_16 = tf.split(d_out[self.resolution_ + '16'][0], 2)
                self.d_loss_real_16 = tf.reduce_mean(tf.abs(AE_x_16 - img_16))
                self.d_loss_fake_16 = tf.reduce_mean(tf.abs(AE_G_16 - g_fake_16))
                self.d_loss_16 = self.d_loss_real_16 - self.k_t_16 * self.d_loss_fake_16
                self.g_loss_16 = tf.reduce_mean(
                    tf.abs(AE_G_16 - g_fake_16)) + self.cfg.lambda_s * self.shape_loss_16 + \
                                 self.cfg.lambda_e * self.expr_loss_16 \
                                 + self.cfg.lambda_p * self.pose_loss_16 + self.cfg.lambda_id * self.id_loss_16

                AE_G_32, AE_x_32 = tf.split(d_out[self.resolution_ + '32'][0], 2)
                self.d_loss_real_32 = tf.reduce_mean(tf.abs(AE_x_32 - img_32))
                self.d_loss_fake_32 = tf.reduce_mean(tf.abs(AE_G_32 - g_fake_32))
                self.d_loss_32 = self.d_loss_real_32 - self.k_t_32 * self.d_loss_fake_32
                self.g_loss_32 = tf.reduce_mean(
                    tf.abs(AE_G_32 - g_fake_32)) + self.cfg.lambda_s * self.shape_loss_32 + \
                                 self.cfg.lambda_e * self.expr_loss_32 \
                                 + self.cfg.lambda_p * self.pose_loss_32 + self.cfg.lambda_id * self.id_loss_32

                AE_G_64, AE_x_64 = tf.split(d_out[self.resolution_ + '64'][0], 2)
                self.d_loss_real_64 = tf.reduce_mean(tf.abs(AE_x_64 - img_64))
                self.d_loss_fake_64 = tf.reduce_mean(tf.abs(AE_G_64 - g_fake_64))
                self.d_loss_64 = self.d_loss_real_64 - self.k_t_64 * self.d_loss_fake_64
                self.g_loss_64 = tf.reduce_mean(
                    tf.abs(AE_G_64 - g_fake_64)) + self.cfg.lambda_s * self.shape_loss_64 + \
                                  self.cfg.lambda_e * self.expr_loss_64 \
                                  + self.cfg.lambda_p * self.pose_loss_64 + self.cfg.lambda_id * self.id_loss_64

                AE_G_128, AE_x_128 = tf.split(d_out[self.resolution_ + '128'][0], 2)
                self.d_loss_real_128 = tf.reduce_mean(tf.abs(AE_x_128 - img_128))
                self.d_loss_fake_128 = tf.reduce_mean(tf.abs(AE_G_128 - g_fake_128))
                self.d_loss_128 = self.d_loss_real_128 - self.k_t_128 * self.d_loss_fake_128
                self.g_loss_128 = tf.reduce_mean(tf.abs(AE_G_128 - g_fake_128)) + self.cfg.lambda_s * self.shape_loss_128 + \
                              self.cfg.lambda_e * self.expr_loss_128 \
                              + self.cfg.lambda_p * self.pose_loss_128 + self.cfg.lambda_id * self.id_loss_128

                tf.summary.image('image_fake_8', g_fake_8, 3)
                tf.summary.scalar('g_loss_8', self.g_loss_8)
                tf.summary.scalar('d_loss_8', self.d_loss_8)
                tf.summary.scalar('shape_loss_8', self.shape_loss_8)
                tf.summary.scalar('expr_loss_8', self.expr_loss_8)
                tf.summary.scalar('pose_loss_8', self.pose_loss_8)
                tf.summary.scalar('id_loss_8', self.id_loss_8)

                tf.summary.image('image_fake_16', g_fake_16, 3)
                tf.summary.scalar('g_loss_16', self.g_loss_16)
                tf.summary.scalar('d_loss_16', self.d_loss_16)
                tf.summary.scalar('shape_loss_16', self.shape_loss_16)
                tf.summary.scalar('expr_loss_16', self.expr_loss_16)
                tf.summary.scalar('pose_loss_16', self.pose_loss_16)
                tf.summary.scalar('id_loss_16', self.id_loss_16)

                tf.summary.image('image_fake_32', g_fake_32, 3)
                tf.summary.scalar('g_loss_32', self.g_loss_32)
                tf.summary.scalar('d_loss_32', self.d_loss_32)
                tf.summary.scalar('shape_loss_32', self.shape_loss_32)
                tf.summary.scalar('expr_loss_32', self.expr_loss_32)
                tf.summary.scalar('pose_loss_32', self.pose_loss_32)
                tf.summary.scalar('id_loss_32', self.id_loss_32)

                tf.summary.image('image_fake_64', g_fake_64, 3)
                tf.summary.scalar('g_loss_64', self.g_loss_64)
                tf.summary.scalar('d_loss_64', self.d_loss_64)
                tf.summary.scalar('shape_loss_64', self.shape_loss_64)
                tf.summary.scalar('expr_loss_64', self.expr_loss_64)
                tf.summary.scalar('pose_loss_64', self.pose_loss_64)
                tf.summary.scalar('id_loss_64', self.id_loss_64)

                tf.summary.image('image_fake_128', g_fake_128, 3)
                tf.summary.image('image_real', img_128, 3)
                tf.summary.scalar('g_loss_128', self.g_loss_128)
                tf.summary.scalar('d_loss_128', self.d_loss_128)
                tf.summary.scalar('shape_loss_128', self.shape_loss_128)
                tf.summary.scalar('expr_loss_128', self.expr_loss_128)
                tf.summary.scalar('pose_loss_128', self.pose_loss_128)
                tf.summary.scalar('id_loss_128', self.id_loss_128)

                grad_g_8 = self.op_g.compute_gradients(self.g_loss_8 * 0.00001, var_list=self.G[self.resolution_ + '8'][1])
                grads_g_8.append(grad_g_8)
                grad_d_8 = self.op_d.compute_gradients(self.d_loss_8 * 0.00001, var_list=d_out[self.resolution_ + '8'][1])
                grads_d_8.append(grad_d_8)

                grad_g_16 = self.op_g.compute_gradients(self.g_loss_16 * 0.0001, var_list=self.G[self.resolution_ + '16'][1])
                grads_g_16.append(grad_g_16)
                grad_d_16 = self.op_d.compute_gradients(self.d_loss_16 * 0.0001, var_list=d_out[self.resolution_ + '16'][1])
                grads_d_16.append(grad_d_16)

                grad_g_32 = self.op_g.compute_gradients(self.g_loss_32 * 0.01, var_list=self.G[self.resolution_ + '32'][1])
                grads_g_32.append(grad_g_32)
                grad_d_32 = self.op_d.compute_gradients(self.d_loss_32 * 0.01, var_list=d_out[self.resolution_ + '32'][1])
                grads_d_32.append(grad_d_32)

                grad_g_64 = self.op_g.compute_gradients(self.g_loss_64 * 0.1, var_list=self.G[self.resolution_ + '64'][1])
                grads_g_64.append(grad_g_64)
                grad_d_64 = self.op_d.compute_gradients(self.d_loss_64 * 0.1, var_list=d_out[self.resolution_ + '64'][1])
                grads_d_64.append(grad_d_64)

                grad_g_128 = self.op_g.compute_gradients(self.g_loss_128, var_list=self.G[self.resolution_ + '128'][1])
                grads_g_128.append(grad_g_128)
                grad_d_128 = self.op_d.compute_gradients(self.d_loss_128, var_list=d_out[self.resolution_ + '128'][1])
                grads_d_128.append(grad_d_128)
            print('Init GPU:{}'.format(i))

        mean_grad_g_8 = m4_average_grads(grads_g_8)
        mean_grad_d_8 = m4_average_grads(grads_d_8)
        self.g_optim_8 = self.op_g.apply_gradients(mean_grad_g_8)
        self.d_optim_8 = self.op_d.apply_gradients(mean_grad_d_8)
        self.balance_8 = self.gamma * self.d_loss_real_8 - self.g_loss_8
        self.measure_8 = self.d_loss_real_8 + tf.abs(self.balance_8)
        tf.summary.scalar('measure_8', self.measure_8)

        mean_grad_g_16 = m4_average_grads(grads_g_16)
        mean_grad_d_16 = m4_average_grads(grads_d_16)
        self.g_optim_16 = self.op_g.apply_gradients(mean_grad_g_16)
        self.d_optim_16 = self.op_d.apply_gradients(mean_grad_d_16)
        self.balance_16 = self.gamma * self.d_loss_real_16 - self.g_loss_16
        self.measure_16 = self.d_loss_real_16 + tf.abs(self.balance_16)
        tf.summary.scalar('measure_16', self.measure_16)

        mean_grad_g_32 = m4_average_grads(grads_g_32)
        mean_grad_d_32 = m4_average_grads(grads_d_32)
        self.g_optim_32 = self.op_g.apply_gradients(mean_grad_g_32)
        self.d_optim_32 = self.op_d.apply_gradients(mean_grad_d_32)
        self.balance_32 = self.gamma * self.d_loss_real_32 - self.g_loss_32
        self.measure_32 = self.d_loss_real_32 + tf.abs(self.balance_32)
        tf.summary.scalar('measure_32', self.measure_32)

        mean_grad_g_64 = m4_average_grads(grads_g_64)
        mean_grad_d_64 = m4_average_grads(grads_d_64)
        self.g_optim_64 = self.op_g.apply_gradients(mean_grad_g_64)
        self.d_optim_64 = self.op_d.apply_gradients(mean_grad_d_64)
        self.balance_64 = self.gamma * self.d_loss_real_64 - self.g_loss_64
        self.measure_64 = self.d_loss_real_64 + tf.abs(self.balance_64)
        tf.summary.scalar('measure_64', self.measure_64)

        mean_grad_g_128 = m4_average_grads(grads_g_128)
        mean_grad_d_128 = m4_average_grads(grads_d_128)
        self.g_optim_128 = self.op_g.apply_gradients(mean_grad_g_128)
        self.d_optim_128 = self.op_d.apply_gradients(mean_grad_d_128, global_step=self.global_step)
        self.balance_128 = self.gamma * self.d_loss_real_128 - self.g_loss_128
        self.measure_128 = self.d_loss_real_128 + tf.abs(self.balance_128)
        tf.summary.scalar('measure_128', self.measure_128)

        self.k_update_8 = tf.assign(self.k_t_8, tf.clip_by_value(self.k_t_8 + self.lambda_k * self.balance_8, 0, 1))
        self.k_update_16 = tf.assign(self.k_t_16, tf.clip_by_value(self.k_t_16 + self.lambda_k * self.balance_16, 0, 1))
        self.k_update_32 = tf.assign(self.k_t_32, tf.clip_by_value(self.k_t_32 + self.lambda_k * self.balance_32, 0, 1))
        self.k_update_64 = tf.assign(self.k_t_64, tf.clip_by_value(self.k_t_64 + self.lambda_k * self.balance_64, 0, 1))

        tf.summary.scalar('kt_8', self.k_t_8)
        tf.summary.scalar('kt_16', self.k_t_16)
        tf.summary.scalar('kt_32', self.k_t_32)
        tf.summary.scalar('kt_64', self.k_t_64)
        tf.summary.scalar('kt_128', self.k_t_128)
        with tf.control_dependencies([self.d_optim_8, self.g_optim_8, self.k_update_8,
                                      self.d_optim_16, self.g_optim_16, self.k_update_16,
                                      self.d_optim_32, self.g_optim_32, self.k_update_32,
                                      self.d_optim_64, self.g_optim_64, self.k_update_64,
                                      self.d_optim_128, self.g_optim_128]):
            self.k_update_128 = tf.assign(self.k_t_128, tf.clip_by_value(self.k_t_128 + self.lambda_k * self.balance_128, 0, 1))


    def build_model_test(self, images, labels, z, id_feat_real, shape_real_norm, expr_real_norm, pose_real_norm):
        _, height, width, self.channel = \
            self.get_conv_shape(images, self.data_format)
        self.repeat_num = int(np.log2(height)) - 2

        self.id_feat_real = self.m4_ID_Extractor(images, reuse=False)
        self.shape_real_norm, self.expr_real_norm, self.pose_real_norm = self.model_3DMM_default_graph(
            self.expr_shape_pose, images,
            reuse=False)
        z_concat_feat = tf.concat([z, shape_real_norm, pose_real_norm, expr_real_norm, id_feat_real], axis=1)
        self.G, self.G_var = self.GeneratorCNN(z_concat_feat, self.conv_hidden_num, self.channel, self.repeat_num,
                                               self.data_format,
                                               reuse=False, name_='generator')
        self.sampler = self.G
        # d_out, self.D_z, self.D_var = self.DiscriminatorCNN(
        #     tf.concat([self.G, images], 0), self.channel, self.z_dim, self.repeat_num,
        #     self.conv_hidden_num, self.data_format,reuse=False, name_='discriminator')
        # AE_G, AE_x = tf.split(d_out, 2)

    def GeneratorCNN(self, z, hidden_num, output_num, repeat_num, data_format, reuse, name_="generator"):
        m4_resolution_dict = defaultdict(list)
        with tf.variable_scope(name_, reuse=reuse) as vs:
            num_output = int(np.prod([8, 8, hidden_num]))
            x = slim.fully_connected(z, num_output, activation_fn=None)
            x = self.reshape(x, 8, 8, hidden_num, data_format)

            for idx in range(repeat_num):
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)

                img_resolution = 8 * (2 ** idx)
                x_resolution = slim.conv2d(x, 3, 3, 1, activation_fn=None, data_format=data_format,
                                           scope=self.resolution_ + str(img_resolution))

                # collect variables
                t_vars = tf.trainable_variables(name_)
                vars_resloution = [var for var in t_vars if
                                   self.resolution_ + str(img_resolution) in var.name or 'Conv' in var.name or 'fully' in var.name]
                # 获取对应分辨率下的输出图像和训练变量

                m4_resolution_dict[self.resolution_ + str(img_resolution)].append(x_resolution)
                m4_resolution_dict[self.resolution_ + str(img_resolution)].append(vars_resloution)
                if idx < repeat_num - 1:
                    # 扩大分辨率
                    x = self.upscale(x, 2, data_format)

            # out = slim.conv2d(x, 3, 3, 1, activation_fn=None, data_format=data_format)

        return m4_resolution_dict

    def DiscriminatorCNN(self, resolution_img_dict, input_channel, z_num, repeat_num, hidden_num, data_format,
                         reuse=False, name_='discriminator'):
        m4_resolution_dict = defaultdict(list)
        x = resolution_img_dict[self.resolution_ + '128'][0]
        # _, height, width, self.channel = self.get_conv_shape(x, self.data_format)
        # resolution_encoder_str = height // 2

        variables_name_list = [self.resolution_ + 'trian_128',
                               self.resolution_ + 'trian_128' + 'trian_64',
                               self.resolution_ + 'trian_128' + 'trian_64' + 'trian_32',
                               self.resolution_ + 'trian_128' + 'trian_64' + 'trian_32' + 'trian_16',
                               self.resolution_ + 'trian_128' + 'trian_64' + 'trian_32' + 'trian_16' + 'trian_8']
        variables_name_list_v = ['trian_8', 'trian_16', 'trian_32', 'trian_64', 'trian_128']

        with tf.variable_scope(name_, reuse=reuse) as vs:
            # Encoder
            x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format, scope=variables_name_list[0])
            # prev_channel_num = hidden_num

            for idx in range(repeat_num):
                channel_num = hidden_num * (idx + 1)
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format,
                                scope=variables_name_list[idx] + str(idx + 1))
                x = slim.conv2d(x, channel_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format,
                                scope=variables_name_list[idx] + str(idx + 2))

                if idx < repeat_num - 1:
                    x = slim.conv2d(x, channel_num, 3, 2, activation_fn=tf.nn.elu, data_format=data_format, scope=variables_name_list[idx+1])


            x = tf.reshape(x, [-1, np.prod([8, 8, channel_num])])
            z = x = slim.fully_connected(x, z_num, activation_fn=None)

            # Decoder
            num_output = int(np.prod([8, 8, hidden_num]))
            x = slim.fully_connected(x, num_output, activation_fn=None)
            x = self.reshape(x, 8, 8, hidden_num, data_format)

            for idx in range(repeat_num):
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)
                x = slim.conv2d(x, hidden_num, 3, 1, activation_fn=tf.nn.elu, data_format=data_format)

                img_resolution = 8 * (2 ** idx)
                x_resolution = slim.conv2d(x, 3, 3, 1, activation_fn=None, data_format=data_format,
                                           scope=self.resolution_ + str(img_resolution))

                # collect variables
                t_vars = tf.trainable_variables(name_)
                vars_resloution = [var for var in t_vars if
                                   self.resolution_ + str(img_resolution) in var.name or
                                   'Conv' in var.name or 'fully' in var.name or variables_name_list_v[idx] in var.name]

                # or
                # self.resolution_ + 'encoder' + str((8 * 2 ** idx)) in var.name

                # 获取对应分辨率下的输出图像和训练变量

                m4_resolution_dict[self.resolution_ + str(img_resolution)].append(x_resolution)
                m4_resolution_dict[self.resolution_ + str(img_resolution)].append(vars_resloution)

                if idx < repeat_num - 1:
                    x = self.upscale(x, 2, data_format)

            # out = slim.conv2d(x, input_channel, 3, 1, activation_fn=None, data_format=data_format)

        # variables = tf.contrib.framework.get_variables(vs)
        return m4_resolution_dict

    def get_conv_shape(self, tensor, data_format):
        shape = self.int_shape(tensor)
        # always return [N, H, W, C]
        if data_format == 'NCHW':
            return [shape[0], shape[2], shape[3], shape[1]]
        elif data_format == 'NHWC':
            return shape

    def upscale(self, x, scale, data_format):
        _, h, w, _ = self.get_conv_shape(x, data_format)
        return self.resize_nearest_neighbor(x, (h * scale, w * scale), data_format)

    def int_shape(self, tensor):
        shape = tensor.get_shape().as_list()
        return [num if num is not None else -1 for num in shape]

    def reshape(self, x, h, w, c, data_format):
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

    def model_3DMM_default_graph(self, batch_size, expr_shape_pose, images, reuse=False):
        expr_shape_pose.extract_PSE_feats(images, batch_size, reuse=reuse)
        fc1ls = expr_shape_pose.fc1ls
        fc1le = expr_shape_pose.fc1le
        pose_model = expr_shape_pose.pose
        shape_norm = tf.nn.l2_normalize(fc1ls, dim=0)
        expr_norm = tf.nn.l2_normalize(fc1le, dim=0)
        pose_norm = tf.nn.l2_normalize(pose_model, dim=0)
        return shape_norm, expr_norm, pose_norm

    def m4_ID_Extractor(self, images, reuse=False):
        images = self.resize_nearest_neighbor(images, (128, 128), self.data_format)
        with tf.variable_scope('facenet', reuse=reuse) as scope:
            network = importlib.import_module('inception_resnet_v1')
            prelogits, _ = network.inference(images, 1.0,
                                             phase_train=False, bottleneck_layer_size=128,
                                             weight_decay=0.0005)
            # logits = slim.fully_connected(prelogits, 10575, activation_fn=None,
            #                               weights_initializer=slim.initializers.xavier_initializer(),
            #                               weights_regularizer=slim.l2_regularizer(0.0000),
            #                               scope='Logits', reuse=reuse)

            embeddings = tf.nn.l2_normalize(prelogits, 1, 1e-10, name='embeddings')  # this is we need id feat
        return embeddings

    def m4_various_resolution_image(self, x):
        _, height, width, self.channel = self.get_conv_shape(x, self.data_format)
        img_128 = x
        img_64 = self.resize_nearest_neighbor(x, (height // 2, width // 2), self.data_format)
        img_32 = self.resize_nearest_neighbor(x, (height // 4, width // 4), self.data_format)
        img_16 = self.resize_nearest_neighbor(x, (height // 8, width // 8), self.data_format)
        img_8 = self.resize_nearest_neighbor(x, (height // 16, width // 16), self.data_format)
        return img_8, img_16, img_32, img_64, img_128
