import tensorflow as tf
import numpy as np
import os
import ops as m4_ops
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

    def build_model(self, images, labels, z_128, z_64, z_32, z_16, z_8):
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

        G_name_dict = {'G8':'G_resolution_8x8', 'G16':'G_resolution_16x16', 'G32':'G_resolution_32x32',
                       'G64': 'G_resolution_64x64', 'G128': 'G_resolution_128x128'}
        D_name_dict = {'D8': 'D_resolution_8x8', 'D16': 'D_resolution_16x16', 'D32': 'D_resolution_32x32',
                       'D64': 'D_resolution_64x64', 'D128': 'D_resolution_128x128'}

        for i in range(self.cfg.num_gpus):
            images_on_one_gpu = images[self.cfg.batch_size * i:self.cfg.batch_size * (i + 1)]
            z_128_on_one_gpu = z_128[self.cfg.batch_size * i:self.cfg.batch_size * (i + 1)]
            z_64_on_one_gpu = z_64[self.cfg.batch_size * i:self.cfg.batch_size * (i + 1)]
            z_32_on_one_gpu = z_32[self.cfg.batch_size * i:self.cfg.batch_size * (i + 1)]
            z_16_on_one_gpu = z_16[self.cfg.batch_size * i:self.cfg.batch_size * (i + 1)]
            z_8_on_one_gpu = z_8[self.cfg.batch_size * i:self.cfg.batch_size * (i + 1)]

            with tf.device("/gpu:{}".format(i)):
                if i == 0:
                    muti_gpu_reuse_0 = False
                    muti_gpu_reuse_1 = True
                else:
                    muti_gpu_reuse_0 = True
                    muti_gpu_reuse_1 = True

                img_8, img_16, img_32, img_64, img_128 = self.m4_various_resolution_image(images_on_one_gpu)
                id_feat_real = self.m4_ID_Extractor(img_128, reuse=muti_gpu_reuse_0)
                shape_real_norm, expr_real_norm, pose_real_norm = self.model_3DMM_default_graph(self.cfg.batch_size,
                                                                                                self.expr_shape_pose,
                                                                                                img_128,
                                                                                                reuse=muti_gpu_reuse_0)
                z_8_concat_feat = tf.concat([z_8_on_one_gpu, shape_real_norm, pose_real_norm,
                                             expr_real_norm, id_feat_real], axis=1)
                G_img_8x8, vars_G_8x8 = self.m4_G_resolution_8x8(z_8_concat_feat, name_dict=G_name_dict,
                                                             reuse_0=muti_gpu_reuse_0, reuse_1=muti_gpu_reuse_1, reso=8,
                                                             hidden_num=self.cfg.conv_hidden_num)

                id_feat_fake_8 = self.m4_ID_Extractor(G_img_8x8, reuse=muti_gpu_reuse_1)
                shape_fake_norm_8, expr_fake_norm_8, pose_fake_norm_8 = self.model_3DMM_default_graph(self.cfg.batch_size,
                                                                                                self.expr_shape_pose,
                                                                                                G_img_8x8,
                                                                                                reuse=muti_gpu_reuse_1)
                z_16_concat_feat = tf.concat([z_16_on_one_gpu, shape_fake_norm_8, pose_fake_norm_8,
                                              expr_fake_norm_8, id_feat_fake_8], axis=1)
                G_img_16x16, vars_G_16x16 = self.m4_G_resolution_16x16(z_16_concat_feat, name_dict=G_name_dict,
                                                                   reuse_0=muti_gpu_reuse_0, reuse_1=muti_gpu_reuse_1, reso=8,
                                                                   hidden_num=self.cfg.conv_hidden_num)

                id_feat_fake_16 = self.m4_ID_Extractor(G_img_16x16, reuse=muti_gpu_reuse_1)
                shape_fake_norm_16, expr_fake_norm_16, pose_fake_norm_16 = self.model_3DMM_default_graph(
                    self.cfg.batch_size,
                    self.expr_shape_pose,
                    G_img_16x16,
                    reuse=muti_gpu_reuse_1)
                z_32_concat_feat = tf.concat([z_32_on_one_gpu, shape_fake_norm_16, pose_fake_norm_16,
                                              expr_fake_norm_16, id_feat_fake_16], axis=1)
                G_img_32x32, vars_G_32x32 = self.m4_G_resolution_32x32(z_32_concat_feat, name_dict=G_name_dict,
                                                                   reuse_0=muti_gpu_reuse_0, reuse_1=muti_gpu_reuse_1,
                                                                   reso=8,
                                                                   hidden_num=self.cfg.conv_hidden_num)

                id_feat_fake_32 = self.m4_ID_Extractor(G_img_32x32, reuse=muti_gpu_reuse_1)
                shape_fake_norm_32, expr_fake_norm_32, pose_fake_norm_32 = self.model_3DMM_default_graph(
                    self.cfg.batch_size,
                    self.expr_shape_pose,
                    G_img_32x32,
                    reuse=muti_gpu_reuse_1)
                z_64_concat_feat = tf.concat([z_64_on_one_gpu, shape_fake_norm_32, pose_fake_norm_32,
                                              expr_fake_norm_32, id_feat_fake_32], axis=1)
                G_img_64x64, vars_G_64x64 = self.m4_G_resolution_64x64(z_64_concat_feat, name_dict=G_name_dict,
                                                                     reuse_0=muti_gpu_reuse_0, reuse_1=muti_gpu_reuse_1,
                                                                     reso=8,
                                                                     hidden_num=self.cfg.conv_hidden_num)

                id_feat_fake_64 = self.m4_ID_Extractor(G_img_64x64, reuse=muti_gpu_reuse_1)
                shape_fake_norm_64, expr_fake_norm_64, pose_fake_norm_64 = self.model_3DMM_default_graph(
                    self.cfg.batch_size,
                    self.expr_shape_pose,
                    G_img_64x64,
                    reuse=muti_gpu_reuse_1)
                z_128_concat_feat = tf.concat([z_128_on_one_gpu, shape_fake_norm_64, pose_fake_norm_64,
                                              expr_fake_norm_64, id_feat_fake_64], axis=1)
                G_img_128x128, vars_G_128x128 = self.m4_G_resolution_128x128(z_128_concat_feat, name_dict=G_name_dict,
                                                                     reuse_0=muti_gpu_reuse_0, reuse_1=muti_gpu_reuse_1,
                                                                     reso=8,
                                                                     hidden_num=self.cfg.conv_hidden_num)

                id_feat_fake_128 = self.m4_ID_Extractor(G_img_128x128, reuse=muti_gpu_reuse_1)
                shape_fake_norm_128, expr_fake_norm_128, pose_fake_norm_128 = self.model_3DMM_default_graph(
                    self.cfg.batch_size,
                    self.expr_shape_pose,
                    G_img_128x128,
                    reuse=muti_gpu_reuse_1)

                if i == 0:
                    self.sampler = G_img_128x128

                AE_G_8x8, AE_D_vars_8x8 = self.m4_D_resolution_8x8(G_img_8x8, D_name_dict, muti_gpu_reuse_0, muti_gpu_reuse_1,
                                                                   reso=8, hidden_num=self.cfg.conv_hidden_num)
                AE_G_16x16, AE_D_vars_16x16 = self.m4_D_resolution_16x16(G_img_16x16, D_name_dict, muti_gpu_reuse_0,
                                                                   muti_gpu_reuse_1,
                                                                   reso=8, hidden_num=self.cfg.conv_hidden_num)
                AE_G_32x32, AE_D_vars_32x32 = self.m4_D_resolution_32x32(G_img_32x32, D_name_dict, muti_gpu_reuse_0,
                                                                         muti_gpu_reuse_1,
                                                                         reso=8, hidden_num=self.cfg.conv_hidden_num)
                AE_G_64x64, AE_D_vars_64x64 = self.m4_D_resolution_64x64(G_img_64x64, D_name_dict, muti_gpu_reuse_0,
                                                                         muti_gpu_reuse_1,
                                                                         reso=8, hidden_num=self.cfg.conv_hidden_num)
                AE_G_128x128, AE_D_vars_128x128 = self.m4_D_resolution_128x128(G_img_128x128, D_name_dict, muti_gpu_reuse_0,
                                                                         muti_gpu_reuse_1,
                                                                         reso=8, hidden_num=self.cfg.conv_hidden_num)

                AE_X_8x8, AE_D_vars_8x8 = self.m4_D_resolution_8x8(img_8, D_name_dict, muti_gpu_reuse_1,
                                                                   muti_gpu_reuse_1,
                                                                   reso=8, hidden_num=self.cfg.conv_hidden_num)
                AE_X_16x16, AE_D_vars_16x16 = self.m4_D_resolution_16x16(img_16, D_name_dict, muti_gpu_reuse_1,
                                                                         muti_gpu_reuse_1,
                                                                         reso=8, hidden_num=self.cfg.conv_hidden_num)
                AE_X_32x32, AE_D_vars_32x32 = self.m4_D_resolution_32x32(img_32, D_name_dict, muti_gpu_reuse_1,
                                                                         muti_gpu_reuse_1,
                                                                         reso=8, hidden_num=self.cfg.conv_hidden_num)
                AE_X_64x64, AE_D_vars_64x64 = self.m4_D_resolution_64x64(img_64, D_name_dict, muti_gpu_reuse_1,
                                                                         muti_gpu_reuse_1,
                                                                         reso=8, hidden_num=self.cfg.conv_hidden_num)
                AE_X_128x128, AE_D_vars_128x128 = self.m4_D_resolution_128x128(img_128, D_name_dict,
                                                                               muti_gpu_reuse_1,
                                                                               muti_gpu_reuse_1,
                                                                               reso=8,
                                                                               hidden_num=self.cfg.conv_hidden_num)

                self.shape_loss_128 = tf.reduce_mean(tf.square(shape_real_norm - shape_fake_norm_128))
                self.expr_loss_128 = tf.reduce_mean(tf.square(expr_real_norm - expr_fake_norm_128))
                self.pose_loss_128 = tf.reduce_mean(tf.square(pose_real_norm - pose_fake_norm_128))
                self.id_loss_128 = tf.reduce_mean(tf.square(id_feat_real - id_feat_fake_128))

                self.shape_loss_64 = tf.reduce_mean(tf.square(shape_real_norm - shape_fake_norm_64))
                self.expr_loss_64 = tf.reduce_mean(tf.square(expr_real_norm - expr_fake_norm_64))
                self.pose_loss_64 = tf.reduce_mean(tf.square(pose_real_norm - pose_fake_norm_64))
                self.id_loss_64 = tf.reduce_mean(tf.square(id_feat_real - id_feat_fake_64))

                # self.shape_loss_32 = tf.reduce_mean(tf.square(shape_real_norm - shape_fake_norm_32))
                # self.expr_loss_32 = tf.reduce_mean(tf.square(expr_real_norm - expr_fake_norm_32))
                # self.pose_loss_32 = tf.reduce_mean(tf.square(pose_real_norm - pose_fake_norm_32))
                # self.id_loss_32 = tf.reduce_mean(tf.square(id_feat_real - id_feat_fake_32))
                #
                # self.shape_loss_16 = tf.reduce_mean(tf.square(shape_real_norm - shape_fake_norm_16))
                # self.expr_loss_16 = tf.reduce_mean(tf.square(expr_real_norm - expr_fake_norm_16))
                # self.pose_loss_16 = tf.reduce_mean(tf.square(pose_real_norm - pose_fake_norm_16))
                # self.id_loss_16 = tf.reduce_mean(tf.square(id_feat_real - id_feat_fake_16))
                #
                # self.shape_loss_8 = tf.reduce_mean(tf.square(shape_real_norm - shape_fake_norm_8))
                # self.expr_loss_8 = tf.reduce_mean(tf.square(expr_real_norm - expr_fake_norm_8))
                # self.pose_loss_8 = tf.reduce_mean(tf.square(pose_real_norm - pose_fake_norm_8))
                # self.id_loss_8 = tf.reduce_mean(tf.square(id_feat_real - id_feat_fake_8))


                # GAN LOSS
                self.d_loss_real_8 = tf.reduce_mean(tf.abs(AE_X_8x8 - img_8))
                self.d_loss_fake_8 = tf.reduce_mean(tf.abs(AE_G_8x8 - G_img_8x8))
                self.d_loss_8 = self.d_loss_real_8 - self.k_t_8 * self.d_loss_fake_8
                self.g_loss_8 = self.d_loss_fake_8


                self.d_loss_real_16 = tf.reduce_mean(tf.abs(AE_X_16x16 - img_16))
                self.d_loss_fake_16 = tf.reduce_mean(tf.abs(AE_G_16x16 - G_img_16x16))
                self.d_loss_16 = self.d_loss_real_16 - self.k_t_16 * self.d_loss_fake_16
                self.g_loss_16 = self.d_loss_fake_16


                self.d_loss_real_32 = tf.reduce_mean(tf.abs(AE_X_32x32 - img_32))
                self.d_loss_fake_32 = tf.reduce_mean(tf.abs(AE_G_32x32 - G_img_32x32))
                self.d_loss_32 = self.d_loss_real_32 - self.k_t_32 * self.d_loss_fake_32
                self.g_loss_32 = self.d_loss_fake_32


                self.d_loss_real_64 = tf.reduce_mean(tf.abs(AE_X_64x64 - img_64))
                self.d_loss_fake_64 = tf.reduce_mean(tf.abs(AE_G_64x64 - G_img_64x64))
                self.d_loss_64 = self.d_loss_real_64 - self.k_t_64 * self.d_loss_fake_64
                self.g_loss_64 = self.d_loss_fake_64


                self.d_loss_real_128 = tf.reduce_mean(tf.abs(AE_X_128x128 - img_128))
                self.d_loss_fake_128 = tf.reduce_mean(tf.abs(AE_G_128x128 - G_img_128x128))
                self.d_loss_128 = self.d_loss_real_128 - self.k_t_128 * self.d_loss_fake_128
                self.g_loss_128 = self.d_loss_fake_128 + self.cfg.lambda_s * self.shape_loss_128 + \
                              self.cfg.lambda_e * self.expr_loss_128 \
                              + self.cfg.lambda_p * self.pose_loss_128 + self.cfg.lambda_id * self.id_loss_128

                tf.summary.image('image_fake_8', G_img_8x8, 3)
                tf.summary.scalar('g_loss_8', self.g_loss_8)
                tf.summary.scalar('d_loss_8', self.d_loss_8)
                # tf.summary.scalar('shape_loss_8', self.shape_loss_8)
                # tf.summary.scalar('expr_loss_8', self.expr_loss_8)
                # tf.summary.scalar('pose_loss_8', self.pose_loss_8)
                # tf.summary.scalar('id_loss_8', self.id_loss_8)

                tf.summary.image('image_fake_16', G_img_16x16, 3)
                tf.summary.scalar('g_loss_16', self.g_loss_16)
                tf.summary.scalar('d_loss_16', self.d_loss_16)
                # tf.summary.scalar('shape_loss_16', self.shape_loss_16)
                # tf.summary.scalar('expr_loss_16', self.expr_loss_16)
                # tf.summary.scalar('pose_loss_16', self.pose_loss_16)
                # tf.summary.scalar('id_loss_16', self.id_loss_16)

                tf.summary.image('image_fake_32', G_img_32x32, 3)
                tf.summary.scalar('g_loss_32', self.g_loss_32)
                tf.summary.scalar('d_loss_32', self.d_loss_32)
                # tf.summary.scalar('shape_loss_32', self.shape_loss_32)
                # tf.summary.scalar('expr_loss_32', self.expr_loss_32)
                # tf.summary.scalar('pose_loss_32', self.pose_loss_32)
                # tf.summary.scalar('id_loss_32', self.id_loss_32)

                tf.summary.image('image_fake_64', G_img_64x64, 3)
                tf.summary.scalar('g_loss_64', self.g_loss_64)
                tf.summary.scalar('d_loss_64', self.d_loss_64)
                # tf.summary.scalar('shape_loss_64', self.shape_loss_64)
                # tf.summary.scalar('expr_loss_64', self.expr_loss_64)
                # tf.summary.scalar('pose_loss_64', self.pose_loss_64)
                # tf.summary.scalar('id_loss_64', self.id_loss_64)

                tf.summary.image('image_fake_128', G_img_128x128, 3)
                tf.summary.image('image_real', img_128, 3)
                tf.summary.scalar('g_loss_128', self.g_loss_128)
                tf.summary.scalar('d_loss_128', self.d_loss_128)
                tf.summary.scalar('shape_loss_128', self.shape_loss_128)
                tf.summary.scalar('expr_loss_128', self.expr_loss_128)
                tf.summary.scalar('pose_loss_128', self.pose_loss_128)
                tf.summary.scalar('id_loss_128', self.id_loss_128)

                grad_g_8 = self.op_g.compute_gradients(self.g_loss_8 * 0.00001, var_list=vars_G_8x8)
                grads_g_8.append(grad_g_8)
                grad_d_8 = self.op_d.compute_gradients(self.d_loss_8 * 0.00001, var_list=AE_D_vars_8x8)
                grads_d_8.append(grad_d_8)

                grad_g_16 = self.op_g.compute_gradients(self.g_loss_16 * 0.0001, var_list=vars_G_16x16)
                grads_g_16.append(grad_g_16)
                grad_d_16 = self.op_d.compute_gradients(self.d_loss_16 * 0.0001, var_list=AE_D_vars_16x16)
                grads_d_16.append(grad_d_16)

                grad_g_32 = self.op_g.compute_gradients(self.g_loss_32 * 0.001, var_list=vars_G_32x32)
                grads_g_32.append(grad_g_32)
                grad_d_32 = self.op_d.compute_gradients(self.d_loss_32 * 0.001, var_list=AE_D_vars_32x32)
                grads_d_32.append(grad_d_32)

                grad_g_64 = self.op_g.compute_gradients(self.g_loss_64 * 0.1, var_list=vars_G_64x64)
                grads_g_64.append(grad_g_64)
                grad_d_64 = self.op_d.compute_gradients(self.d_loss_64 * 0.1, var_list=AE_D_vars_64x64)
                grads_d_64.append(grad_d_64)

                grad_g_128 = self.op_g.compute_gradients(self.g_loss_128, var_list=vars_G_128x128)
                grads_g_128.append(grad_g_128)
                grad_d_128 = self.op_d.compute_gradients(self.d_loss_128, var_list=AE_D_vars_128x128)
                grads_d_128.append(grad_d_128)
            print('Init GPU:{}'.format(i))

        mean_grad_g_8 = m4_ops.m4_average_grads(grads_g_8)
        mean_grad_d_8 = m4_ops.m4_average_grads(grads_d_8)
        self.g_optim_8 = self.op_g.apply_gradients(mean_grad_g_8)
        self.d_optim_8 = self.op_d.apply_gradients(mean_grad_d_8)
        self.balance_8 = self.gamma * self.d_loss_real_8 - self.g_loss_8
        self.measure_8 = self.d_loss_real_8 + tf.abs(self.balance_8)
        tf.summary.scalar('measure_8', self.measure_8)

        mean_grad_g_16 = m4_ops.m4_average_grads(grads_g_16)
        mean_grad_d_16 = m4_ops.m4_average_grads(grads_d_16)
        self.g_optim_16 = self.op_g.apply_gradients(mean_grad_g_16)
        self.d_optim_16 = self.op_d.apply_gradients(mean_grad_d_16)
        self.balance_16 = self.gamma * self.d_loss_real_16 - self.g_loss_16
        self.measure_16 = self.d_loss_real_16 + tf.abs(self.balance_16)
        tf.summary.scalar('measure_16', self.measure_16)

        mean_grad_g_32 = m4_ops.m4_average_grads(grads_g_32)
        mean_grad_d_32 = m4_ops.m4_average_grads(grads_d_32)
        self.g_optim_32 = self.op_g.apply_gradients(mean_grad_g_32)
        self.d_optim_32 = self.op_d.apply_gradients(mean_grad_d_32)
        self.balance_32 = self.gamma * self.d_loss_real_32 - self.g_loss_32
        self.measure_32 = self.d_loss_real_32 + tf.abs(self.balance_32)
        tf.summary.scalar('measure_32', self.measure_32)

        mean_grad_g_64 = m4_ops.m4_average_grads(grads_g_64)
        mean_grad_d_64 = m4_ops.m4_average_grads(grads_d_64)
        self.g_optim_64 = self.op_g.apply_gradients(mean_grad_g_64)
        self.d_optim_64 = self.op_d.apply_gradients(mean_grad_d_64)
        self.balance_64 = self.gamma * self.d_loss_real_64 - self.g_loss_64
        self.measure_64 = self.d_loss_real_64 + tf.abs(self.balance_64)
        tf.summary.scalar('measure_64', self.measure_64)

        mean_grad_g_128 = m4_ops.m4_average_grads(grads_g_128)
        mean_grad_d_128 = m4_ops.m4_average_grads(grads_d_128)
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







    def m4_conv_moudel(self, x, fiters, k_h=3, k_w=3, s_h=1, s_w=1, padding="SAME", stddev=0.02,
                    active_function='elu', reuse= False, name='conv_moudel'):
        with tf.variable_scope(name, reuse=reuse) as scope:
            x, vars = m4_ops.m4_conv(x, fiters, k_h=k_h, k_w=k_w, s_h=s_h, s_w=s_w,
                               padding=padding, stddev=stddev, name='conv')
            x = m4_ops.m4_active_function(x, active_function=active_function, name='active_function')

            x, vars1 = m4_ops.m4_conv(x, fiters, k_h=k_h, k_w=k_w, s_h=s_h, s_w=s_w,
                               padding=padding, stddev=stddev, name='conv1')
            x = m4_ops.m4_active_function(x, active_function=active_function, name='active_function1')
            vars_all = tf.contrib.framework.get_variables(scope)
            return x, vars_all

    def m4_downscale_conv_moudel(self, x, fiters, k_h=3, k_w=3, s_h=2, s_w=2, padding="SAME", stddev=0.02,
                    active_function='elu', reuse= False, name='downscale_conv_moudel'):
        with tf.variable_scope(name, reuse=reuse) as scope:
            x, vars = m4_ops.m4_conv(x, fiters, k_h=k_h, k_w=k_w, s_h=s_h, s_w=s_w,
                               padding=padding, stddev=stddev, name='conv')
            x = m4_ops.m4_active_function(x, active_function=active_function, name='active_function')
            vars_all = tf.contrib.framework.get_variables(scope)
            return x, vars_all

    def m4_reshape_ftoc(self,x, h, w, nc):
        x = tf.reshape(x, [-1, h, w, nc], name='reshape_ftoc')
        return x

    def m4_FullConnect(self,x, output, active_function=None, stddev=0.02, reuse=False, name='FC'):
        with tf.variable_scope(name, reuse=reuse) as scope:
            x = m4_ops.m4_FullConnect(x, output, active_function=active_function, stddev=stddev, reuse=reuse, name='fullconnect')
            vars = tf.contrib.framework.get_variables(scope)
            return x, vars

    def m4_to_RGB(self, x, fiters, k_h=3, k_w=3, s_h=1, s_w=1,
                               padding='SAME', stddev=0.02, reuse=False, name='to_GRB'):
        with tf.variable_scope(name, reuse=reuse) as scope:
            x, vars = m4_ops.m4_conv(x, fiters, k_h=k_h, k_w=k_w, s_h=s_h, s_w=s_w,
                                   padding=padding, stddev=stddev, name='conv')
            return x, vars

    def m4_from_RGB(self, x, fiters, k_h=3, k_w=3, s_h=1, s_w=1,
                               padding='SAME', stddev=0.02, reuse=False, name='from_RGB'):
        with tf.variable_scope(name, reuse=reuse) as scope:
            x, vars = m4_ops.m4_conv(x, fiters, k_h=k_h, k_w=k_w, s_h=s_h, s_w=s_w,
                                   padding=padding, stddev=stddev, name='conv')
            return x, vars

    def m4_G_resolution_8x8(self, z_feat, name_dict, reuse_0, reuse_1, reso=8, hidden_num=128):
        x, vars_FC = self.m4_FullConnect(z_feat, reso * reso * hidden_num, active_function=None, reuse=reuse_0, name='FC8')
        x = self.m4_reshape_ftoc(x, reso, reso, hidden_num)
        x, vars_8x8 = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_0, name=name_dict['G8'])
        to_GRB, vars_to_GRB_8x8 = self.m4_to_RGB(x, 3, reuse=reuse_0, name="to_RGB_8x8")
        vars = vars_FC + vars_8x8 + vars_to_GRB_8x8
        return to_GRB, vars

    def m4_G_resolution_16x16(self, z_feat, name_dict, reuse_0, reuse_1, reso=8, hidden_num=128):
        x, vars_FC = self.m4_FullConnect(z_feat, reso*reso* hidden_num, active_function=None, reuse=reuse_0, name='FC16')
        x = self.m4_reshape_ftoc(x, reso, reso, hidden_num)

        x, vars_8x8 = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_1, name=name_dict['G8'])
        x = tf.image.resize_nearest_neighbor(x, (16, 16))
        x, vars_16x16 = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_0, name=name_dict['G16'])
        to_GRB, vars_to_GRB_16x16 = self.m4_to_RGB(x, 3, reuse=reuse_0, name="to_RGB_16x16")
        vars = vars_FC + vars_8x8 + vars_16x16 + vars_to_GRB_16x16
        return to_GRB, vars

    def m4_G_resolution_32x32(self, z_feat, name_dict, reuse_0, reuse_1, reso=8, hidden_num=128):
        x, vars_FC = self.m4_FullConnect(z_feat, reso*reso* hidden_num, active_function=None, reuse=reuse_0, name='FC32')
        x = self.m4_reshape_ftoc(x, reso, reso, hidden_num)
        x, vars_8x8 = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_1, name=name_dict['G8'])
        x = tf.image.resize_nearest_neighbor(x, (16, 16))
        x, vars_16x16 = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_1, name=name_dict['G16'])
        x = tf.image.resize_nearest_neighbor(x, (32, 32))
        x, vars_32x32 = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_0, name=name_dict['G32'])
        to_GRB, vars_to_GRB_32x32 = self.m4_to_RGB(x, 3, reuse=reuse_0, name="to_RGB_32x32")
        vars = vars_FC + vars_8x8 + vars_16x16 + vars_32x32 + vars_to_GRB_32x32
        return to_GRB, vars

    def m4_G_resolution_64x64(self, z_feat, name_dict, reuse_0, reuse_1, reso=8, hidden_num=128):
        x, vars_FC = self.m4_FullConnect(z_feat, reso*reso* hidden_num, active_function=None, reuse=reuse_0, name='FC64')
        x = self.m4_reshape_ftoc(x, reso, reso, hidden_num)
        x, vars_8x8 = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_1, name=name_dict['G8'])
        x = tf.image.resize_nearest_neighbor(x, (16, 16))
        x, vars_16x16 = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_1, name=name_dict['G16'])
        x = tf.image.resize_nearest_neighbor(x, (32, 32))
        x, vars_32x32 = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_1, name=name_dict['G32'])
        x = tf.image.resize_nearest_neighbor(x, (64, 64))
        x, vars_64x64 = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse=reuse_0, name=name_dict['G64'])
        to_GRB, vars_to_GRB_64x64 = self.m4_to_RGB(x, 3, reuse=reuse_0, name="to_RGB_64x64")
        vars = vars_FC + vars_8x8 + vars_16x16 + vars_32x32 + vars_64x64 + vars_to_GRB_64x64
        return to_GRB, vars

    def m4_G_resolution_128x128(self, z_feat, name_dict, reuse_0, reuse_1, reso=8, hidden_num=128):
        x, vars_FC = self.m4_FullConnect(z_feat, reso*reso* hidden_num, active_function=None, reuse=reuse_0, name='FC128')
        x = self.m4_reshape_ftoc(x, reso, reso, hidden_num)
        x, vars_8x8 = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_1, name=name_dict['G8'])
        x = tf.image.resize_nearest_neighbor(x, (16, 16))
        x, vars_16x16 = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_1, name=name_dict['G16'])
        x = tf.image.resize_nearest_neighbor(x, (32, 32))
        x, vars_32x32 = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_1, name=name_dict['G32'])
        x = tf.image.resize_nearest_neighbor(x, (64, 64))
        x, vars_64x64 = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_1, name=name_dict['G64'])
        x = tf.image.resize_nearest_neighbor(x, (128, 128))
        x, vars_128x128 = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_0, name=name_dict['G128'])
        to_GRB, vars_to_GRB_128x128 = self.m4_to_RGB(x, 3, reuse=reuse_0, name="to_RGB_128x128")
        vars = vars_FC + vars_8x8 + vars_16x16 + vars_32x32 + vars_64x64 + vars_128x128 + vars_to_GRB_128x128
        return to_GRB, vars


    def m4_D_resolution_8x8(self, x, name_dict, reuse_0, reuse_1, reso=8, hidden_num=128):
        from_GRB, vars_from_GRB_8x8 = self.m4_to_RGB(x, hidden_num * 4, reuse=reuse_0, name="from_RGB_8x8")

        x, vars_8x8 = self.m4_conv_moudel(from_GRB, fiters=hidden_num * 5, active_function='elu', reuse=reuse_0,
                                          name=name_dict['D8'])
        x = tf.reshape(x, [-1, np.prod([reso,reso,hidden_num * 5])])
        x, vars_FC1 = self.m4_FullConnect(x, 128, active_function=None, reuse=reuse_0,
                                         name='AE_FC1')
        x, vars_FC2 = self.m4_FullConnect(x, reso*reso*hidden_num, active_function=None, reuse=reuse_0,
                                         name='AE_FC2')
        x = tf.reshape(x, [-1, 8, 8, hidden_num])

        x, vars_8x8_E = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_0, name=name_dict['D8'] + 'D')
        to_GRB, vars_to_GRB_8x8 = self.m4_to_RGB(x, 3, reuse=reuse_0, name="to_RGB_8x8_D")
        vars = vars_from_GRB_8x8 + vars_8x8 + vars_FC1 + vars_FC2 + vars_8x8_E + vars_to_GRB_8x8
        return to_GRB, vars

    def m4_D_resolution_16x16(self, x, name_dict, reuse_0, reuse_1, reso=8, hidden_num=128):
        from_GRB, vars_from_GRB_16x16 = self.m4_to_RGB(x, hidden_num * 3, reuse=reuse_0, name="from_RGB_16x16")
        x, vars_16x16 = self.m4_conv_moudel(from_GRB, fiters=hidden_num * 4, active_function='elu', reuse=reuse_0,
                                          name=name_dict['D16'])
        x, vars_downscale = self.m4_downscale_conv_moudel(x, fiters=hidden_num * 4, active_function='elu', reuse=reuse_0,
                                          name='downscale_16')
        x, vars_8x8 = self.m4_conv_moudel(x, fiters=hidden_num * 5, active_function='elu', reuse=reuse_1,
                                          name=name_dict['D8'])
        x = tf.reshape(x, [-1, np.prod([reso,reso,hidden_num * 5])])
        x, vars_FC1 = self.m4_FullConnect(x, 128, active_function=None, reuse=reuse_1,
                                         name='AE_FC1')
        x, vars_FC2 = self.m4_FullConnect(x, reso*reso*hidden_num, active_function=None, reuse=reuse_1,
                                         name='AE_FC2')
        x = tf.reshape(x, [-1, 8, 8, hidden_num])

        x, vars_8x8_E = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_1, name=name_dict['D8'] + 'D')

        x = tf.image.resize_nearest_neighbor(x, (16, 16))
        x, vars_16x16_E = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse=reuse_0,
                                            name=name_dict['D16'] + 'D')
        to_GRB, vars_to_GRB_16x16 = self.m4_to_RGB(x, 3, reuse=reuse_0, name="to_RGB_16x16_D")
        vars = vars_from_GRB_16x16 + vars_16x16 + vars_downscale + vars_8x8 + \
               vars_FC1 + vars_FC2 + vars_8x8_E + vars_16x16_E + vars_to_GRB_16x16
        return to_GRB, vars

    def m4_D_resolution_32x32(self, x, name_dict, reuse_0, reuse_1, reso=8, hidden_num=128):

        from_GRB, vars_from_GRB_32x32 = self.m4_to_RGB(x, hidden_num * 2, reuse=reuse_0, name="from_RGB_32x32")
        x, vars_32x32 = self.m4_conv_moudel(from_GRB, fiters=hidden_num * 3, active_function='elu', reuse=reuse_0,
                                          name=name_dict['D32'])
        x, vars_downscale32 = self.m4_downscale_conv_moudel(x, fiters=hidden_num * 3, active_function='elu', reuse=reuse_0,
                                          name='downscale_32')
        x, vars_16x16 = self.m4_conv_moudel(x, fiters=hidden_num * 4, active_function='elu', reuse=reuse_1,
                                            name=name_dict['D16'])
        x, vars_downscale16 = self.m4_downscale_conv_moudel(x, fiters=hidden_num * 4, active_function='elu',
                                                          reuse=reuse_1,
                                                          name='downscale_16')
        x, vars_8x8 = self.m4_conv_moudel(x, fiters=hidden_num * 5, active_function='elu', reuse=reuse_1,
                                          name=name_dict['D8'])
        x = tf.reshape(x, [-1, np.prod([reso,reso,hidden_num * 5])])
        x, vars_FC1 = self.m4_FullConnect(x, 128, active_function=None, reuse=reuse_1,
                                         name='AE_FC1')
        x, vars_FC2 = self.m4_FullConnect(x, reso*reso*hidden_num, active_function=None, reuse=reuse_1,
                                         name='AE_FC2')
        x = tf.reshape(x, [-1, 8, 8, hidden_num])

        x, vars_8x8_E = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_1, name=name_dict['D8'] + 'D')

        x = tf.image.resize_nearest_neighbor(x, (16, 16))
        x, vars_16x16_E = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse=reuse_1,
                                            name=name_dict['D16'] + 'D')
        x = tf.image.resize_nearest_neighbor(x, (32, 32))
        x, vars_32x32_E = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_0, name=name_dict['D32'] + 'D')
        to_GRB, vars_to_GRB_32x32 = self.m4_to_RGB(x, 3, reuse=reuse_0, name="to_RGB_32x32_D")
        vars = vars_from_GRB_32x32 + vars_32x32 + vars_downscale32 + vars_16x16 + vars_downscale16 + vars_8x8 + \
               vars_FC1 + vars_FC2 + vars_8x8_E + vars_16x16_E + vars_32x32_E + vars_to_GRB_32x32
        return to_GRB, vars

    def m4_D_resolution_64x64(self, x, name_dict, reuse_0, reuse_1, reso=8, hidden_num=128):
        from_GRB, vars_from_GRB_64x64 = self.m4_to_RGB(x, hidden_num * 1, reuse=reuse_0, name="from_RGB_64x64")
        x, vars_64x64 = self.m4_conv_moudel(from_GRB, fiters=hidden_num * 2, active_function='elu', reuse=reuse_0,
                                            name=name_dict['D64'])
        x, vars_downscale64 = self.m4_downscale_conv_moudel(x, fiters=hidden_num * 2, active_function='elu',
                                                            reuse=reuse_0,
                                                            name='downscale_64')
        x, vars_32x32 = self.m4_conv_moudel(x, fiters=hidden_num * 3, active_function='elu', reuse=reuse_1,
                                          name=name_dict['D32'])
        x, vars_downscale32 = self.m4_downscale_conv_moudel(x, fiters=hidden_num * 3, active_function='elu', reuse=reuse_1,
                                          name='downscale_32')
        x, vars_16x16 = self.m4_conv_moudel(x, fiters=hidden_num * 4, active_function='elu', reuse=reuse_1,
                                            name=name_dict['D16'])
        x, vars_downscale16 = self.m4_downscale_conv_moudel(x, fiters=hidden_num * 4, active_function='elu',
                                                          reuse=reuse_1,
                                                          name='downscale_16')
        x, vars_8x8 = self.m4_conv_moudel(x, fiters=hidden_num * 5, active_function='elu', reuse=reuse_1,
                                          name=name_dict['D8'])
        x = tf.reshape(x, [-1, np.prod([reso,reso,hidden_num * 5])])
        x, vars_FC1 = self.m4_FullConnect(x, 128, active_function=None, reuse=reuse_1,
                                         name='AE_FC1')
        x, vars_FC2 = self.m4_FullConnect(x, reso*reso*hidden_num, active_function=None, reuse=reuse_1,
                                         name='AE_FC2')
        x = tf.reshape(x, [-1, 8, 8, hidden_num])

        x, vars_8x8_E = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_1, name=name_dict['D8'] + 'D')

        x = tf.image.resize_nearest_neighbor(x, (16, 16))
        x, vars_16x16_E = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse=reuse_1,
                                            name=name_dict['D16'] + 'D')
        x = tf.image.resize_nearest_neighbor(x, (32, 32))
        x, vars_32x32_E = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_1, name=name_dict['D32'] + 'D')

        x = tf.image.resize_nearest_neighbor(x, (64, 64))
        x, vars_64x64_E = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse=reuse_0,
                                            name=name_dict['D64'] + 'D')
        to_GRB, vars_to_GRB_64x64 = self.m4_to_RGB(x, 3, reuse=reuse_0, name="to_RGB_64x64_D")

        vars = vars_from_GRB_64x64 + vars_64x64 + vars_downscale64 + vars_32x32 + vars_downscale32 + vars_16x16 + vars_downscale16 + vars_8x8 + \
               vars_FC1 + vars_FC2 + vars_8x8_E + vars_16x16_E + vars_32x32_E + vars_64x64_E + vars_to_GRB_64x64
        return to_GRB, vars

    def m4_D_resolution_128x128(self, x, name_dict, reuse_0, reuse_1, reso=8, hidden_num=128):
        x, vars_128x128 = self.m4_conv_moudel(x, fiters=hidden_num , active_function='elu', reuse=reuse_0,
                                            name=name_dict['D128'])
        x, vars_downscale128 = self.m4_downscale_conv_moudel(x, fiters=hidden_num, active_function='elu',
                                                            reuse=reuse_0,
                                                            name='downscale_128')
        x, vars_64x64 = self.m4_conv_moudel(x, fiters=hidden_num * 2, active_function='elu', reuse=reuse_1,
                                            name=name_dict['D64'])
        x, vars_downscale64 = self.m4_downscale_conv_moudel(x, fiters=hidden_num * 2, active_function='elu',
                                                            reuse=reuse_1,
                                                            name='downscale_64')
        x, vars_32x32 = self.m4_conv_moudel(x, fiters=hidden_num * 3, active_function='elu', reuse=reuse_1,
                                          name=name_dict['D32'])
        x, vars_downscale32 = self.m4_downscale_conv_moudel(x, fiters=hidden_num * 3, active_function='elu', reuse=reuse_1,
                                          name='downscale_32')
        x, vars_16x16 = self.m4_conv_moudel(x, fiters=hidden_num * 4, active_function='elu', reuse=reuse_1,
                                            name=name_dict['D16'])
        x, vars_downscale16 = self.m4_downscale_conv_moudel(x, fiters=hidden_num * 4, active_function='elu',
                                                          reuse=reuse_1,
                                                          name='downscale_16')
        x, vars_8x8 = self.m4_conv_moudel(x, fiters=hidden_num * 5, active_function='elu', reuse=reuse_1,
                                          name=name_dict['D8'])
        x = tf.reshape(x, [-1, np.prod([reso,reso,hidden_num * 5])])
        x, vars_FC1 = self.m4_FullConnect(x, 128, active_function=None, reuse=reuse_1,
                                         name='AE_FC1')
        x, vars_FC2 = self.m4_FullConnect(x, reso*reso*hidden_num, active_function=None, reuse=reuse_1,
                                         name='AE_FC2')
        x = tf.reshape(x, [-1, 8, 8, hidden_num])

        x, vars_8x8_E = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_1, name=name_dict['D8'] + 'D')

        x = tf.image.resize_nearest_neighbor(x, (16, 16))
        x, vars_16x16_E = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse=reuse_1,
                                            name=name_dict['D16'] + 'D')
        x = tf.image.resize_nearest_neighbor(x, (32, 32))
        x, vars_32x32_E = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse= reuse_1, name=name_dict['D32'] + 'D')

        x = tf.image.resize_nearest_neighbor(x, (64, 64))
        x, vars_64x64_E = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse=reuse_1,
                                            name=name_dict['D64'] + 'D')
        x = tf.image.resize_nearest_neighbor(x, (128, 128))
        x, vars_128x128_E = self.m4_conv_moudel(x, fiters=hidden_num, active_function='elu', reuse=reuse_0,
                                              name=name_dict['D128'] + 'D')
        to_GRB, vars_to_GRB_128x128 = self.m4_to_RGB(x, 3, reuse=reuse_0, name="to_RGB_128x128" + "D")
        vars = vars_128x128 + vars_downscale128 + vars_64x64 + vars_downscale64 + vars_32x32 + vars_downscale32 + vars_16x16 + \
               vars_downscale16 + vars_8x8 + \
               vars_FC1 + vars_FC2 + vars_8x8_E + vars_16x16_E + vars_32x32_E + vars_64x64_E + vars_128x128_E + vars_to_GRB_128x128
        return to_GRB, vars

    # def m4_Generator(self, z_dict, name_dict, reuse_0, reuse_1, reso=8, hidden_num=128, reuse=False):
    #     with tf.variable_scope('m4_Generator', reuse=reuse) as scope:
    #         img_8x8, vars_8x8 = self.m4_G_resolution_8x8(z_dict['z8'], name_dict=name_dict,
    #                                                      reuse_0=reuse_0, reuse_1=reuse_1, reso=reso, hidden_num=hidden_num)
    #         img_16x16, vars_16x16 = self.m4_G_resolution_16x16(z_dict['z16'], name_dict=name_dict,
    #                                                            reuse_0=reuse_0, reuse_1=reuse_1, reso=reso, hidden_num=hidden_num)
    #         img_32x32, vars_32x32 = self.m4_G_resolution_32x32(z_dict['z32'], name_dict=name_dict,
    #                                                            reuse_0=reuse_0, reuse_1=reuse_1, reso=reso, hidden_num=hidden_num)
    #         img_64x64, vars_64x64 = self.m4_G_resolution_64x64(z_dict['z64'], name_dict=name_dict,
    #                                                            reuse_0=reuse_0, reuse_1=reuse_1, reso=reso, hidden_num=hidden_num)
    #         img_128x128, vars_128x128 = self.m4_G_resolution_128x128(z_dict['z128'], name_dict=name_dict,
    #                                                                  reuse_0=reuse_0, reuse_1=reuse_1, reso=reso, hidden_num=hidden_num)
    #         return {'G8':[img_8x8, vars_8x8], 'G16':[img_16x16, vars_16x16], 'G32':[img_32x32, vars_32x32],
    #                 'G64':[img_64x64, vars_64x64], 'G128':[img_128x128, vars_128x128]}


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
