from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import os
import datetime
from utils import *
from ops import *
from networks import *
import time
import ExpShapePoseNet as ESP
import scipy
import scipy.io as sio
import utils_3DMM


import argparse

import param

import time

# data_dir = '/media/yang/F/DataSet/Face'
# data_set_name = 'CASIA-WebFace'
# label_dir = '/media/yang/F/DataSet/Face/Label'
# label_name = 'pair_FGLFW.txt'
#
# aaaa = m4_face_label_maker(os.path.join(data_dir,data_set_name),'/media/yang/F/1.txt')
#
# ad = np.loadtxt('/media/yang/F/1.txt',dtype=str)
# print(ad.shape)

class my_gan:
    def __init__(self, sess, cfg):
        self.sess = sess
        self.cfg = cfg
        self.images = tf.placeholder(dtype=tf.float32, shape=[self.cfg.batch_size * self.cfg.num_gpus, 256, 256, 3],
                                     name='real_image')


        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        expr_shape_pose = ESP.m4_3DMM(self.sess, self.cfg)
        expr_shape_pose.extract_PSE_feats(self.images)
        self.fc1ls = expr_shape_pose.fc1ls
        self.fc1le = expr_shape_pose.fc1le
        self.pose_model = expr_shape_pose.pose

    def ESP_test(self):


        # could_load, counter = self.load(self.cfg.checkpoint_dir, self.cfg.dataset_name)
        # if could_load:
        #     print(" [*] Load SUCCESS")
        # else:
        #     print(" [!] Load failed...")
        #
        # names = np.loadtxt(os.path.join(self.cfg.datalabel_dir, self.cfg.datalabel_name), dtype=np.str)
        # dataset_size = names.shape[0]
        # names, labels = m4_get_file_label_name(os.path.join(self.cfg.datalabel_dir, self.cfg.datalabel_name),
        #                                        os.path.join(self.cfg.dataset_dir, self.cfg.dataset_name))
        # filenames = tf.constant(names)
        # filelabels = tf.constant(labels)
        # try:
        #     dataset = tf.data.Dataset.from_tensor_slices((filenames, filelabels))
        # except:
        #     dataset = tf.contrib.data.Dataset.from_tensor_slices((filenames, filelabels))
        #
        # dataset = dataset.map(m4_parse_function)
        # dataset = dataset.shuffle(buffer_size=10000).batch(self.cfg.batch_size * self.cfg.num_gpus).repeat(
        #     self.cfg.epoch)
        # iterator = dataset.make_one_shot_iterator()
        # one_element = iterator.get_next()
        # batch_idxs = dataset_size // (self.cfg.batch_size * self.cfg.num_gpus)
        # batch_images_G, batch_labels_G = self.sess.run(one_element)
        # batch_z_G = np.random.uniform(-1, 1, [self.cfg.batch_size * self.cfg.num_gpus, self.cfg.z_dim]).astype(
        #     np.float32)
        # m4_image_save_cv(batch_images_G,
        #                  '{}/x_fixed.jpg'.format(self.cfg.mesh_folder))
        # print('save x_fixed.jpg.')

        if not os.path.exists(self.cfg.mesh_folder):
            os.makedirs(self.cfg.mesh_folder)



        # x = tf.placeholder(tf.float32, [self.cfg.batch_size * self.cfg.num_gpus, 256, 256, 3])



        print('> Start to estimate Expression, Shape, and Pose!')

        image = cv2.imread('/home/yang/My_Job/study/Gan_Network/BE_GAN_MutiGPU_With_ID/subject1_a.jpg', 1)  # BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_CUBIC)
        image_size_h, image_size_w, nc = image.shape
        image = image / 127.5 - 1.0

        image1 = cv2.imread('/home/yang/My_Job/study/Gan_Network/BE_GAN_MutiGPU_With_ID/subject15_a.jpg', 1)  # BGR
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1 = cv2.resize(image1,(256,256),interpolation=cv2.INTER_CUBIC)
        image_size_h, image_size_w, nc = image1.shape
        image1 = image1 / 127.5 - 1.0

        image_list = []
        image_list.append(image)
        image_list.append(image1)

        image_np = np.asarray(image_list)
        image_np = np.reshape(image_np, [self.cfg.batch_size * 2, image_size_h, image_size_w, 3])

        (Shape_Texture, Expr, Pose) = self.sess.run([self.fc1ls, self.fc1le, self.pose_model], feed_dict={self.images: image_np})
        print(Shape_Texture)
        # -------------------------------make .ply file---------------------------------
        ## Modifed Basel Face Model
        BFM_path = self.cfg.BaselFaceModel_mod_file_path
        model = scipy.io.loadmat(BFM_path, squeeze_me=True, struct_as_record=False)
        model = model["BFM"]
        faces = model.faces - 1
        print('> Loaded the Basel Face Model to write the 3D output!')

        for i in range(self.cfg.batch_size * self.cfg.num_gpus):
            outFile = self.cfg.mesh_folder + '/' + 'haha' + '_' + str(i)

            Pose[i] = np.reshape(Pose[i], [-1])
            Shape_Texture[i] = np.reshape(Shape_Texture[i], [-1])
            Shape = Shape_Texture[i][0:99]
            Shape = np.reshape(Shape, [-1])
            Expr[i] = np.reshape(Expr[i], [-1])

            #########################################
            ### Save 3D shape information (.ply file)

            # Shape + Expression + Pose
            SEP, TEP = utils_3DMM.projectBackBFM_withEP(model, Shape_Texture[i], Expr[i], Pose[i])
            utils_3DMM.write_ply_textureless(outFile + '_Shape_Expr_Pose.ply', SEP, faces)

    # def save(self, checkpoint_dir, step, model_file_name):
    #     model_name = "GAN.model"
    #     checkpoint_dir = os.path.join(checkpoint_dir, model_file_name)
    #
    #     if not os.path.exists(checkpoint_dir):
    #         os.makedirs(checkpoint_dir)
    #
    #     self.saver.save(self.sess,
    #                     os.path.join(checkpoint_dir, model_name),
    #                     global_step=step)
    #
    # def load(self, checkpoint_dir, model_folder_name):
    #     import re
    #     print(" [*] Reading checkpoints...")
    #     checkpoint_dir = os.path.join(checkpoint_dir, model_folder_name)
    #
    #     ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    #     if ckpt and ckpt.model_checkpoint_path:
    #         ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
    #         self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
    #         counter = int(next(re.finditer("(\d+)(?!.*\d)", ckpt_name)).group(0))
    #         print(" [*] Success to read {}".format(ckpt_name))
    #         return True, counter
    #     else:
    #         print(" [*] Failed to find a checkpoint")
    #         return False, 0




os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'  # 指定第  块GPU可用

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
# TF_CPP_MIN_LOG_LEVEL 取值 0 ： 0也是默认值，输出所有信息
# TF_CPP_MIN_LOG_LEVEL 取值 1 ： 屏蔽通知信息
# TF_CPP_MIN_LOG_LEVEL 取值 2 ： 屏蔽通知信息和警告信息
# TF_CPP_MIN_LOG_LEVEL 取值 3 ： 屏蔽通知信息、警告信息和报错信息


parser = argparse.ArgumentParser()
'''
#-----------------------------m4_gan_network-----------------------------
parser.add_argument("--dataset_dir", default=param.dataset_dir, type=str, help="Train data set dir")
parser.add_argument("--dataset_name", default=param.dataset_name, type=str, help="Train data set name")
parser.add_argument("--datalabel_dir", default=param.datalabel_dir, type=str, help="Train data label dir")
parser.add_argument("--datalabel_name", default=param.datalabel_name, type=str, help="Train data label name")
parser.add_argument("--log_dir", default=param.log_dir, type=str, help="Train data label name")
parser.add_argument("--sampel_save_dir", default=param.sampel_save_dir, type=str, help="sampel save dir")
parser.add_argument("--num_gpus", default=param.num_gpus, type=int, help="num of gpu")
parser.add_argument("--epoch", default=param.epoch, type=int, help="epoch")
parser.add_argument("--batch_size", default=param.batch_size, type=int, help="batch size for one gpus")
parser.add_argument("--z_dim", default=param.z_dim, type=int, help="dim of noise")
parser.add_argument("--g_feats", default=param.g_feats, type=int, help="feats for generator")
parser.add_argument("--learning_rate", default=param.learning_rate, type=float, help="learn_rate")
parser.add_argument("--beta1", default=param.beta1, type=float, help="beta1")
parser.add_argument("--beta2", default=param.beta2, type=float, help="beta2")
parser.add_argument("--saveimage_period", default=param.saveimage_period, type=int, help="saveimage_period")
parser.add_argument("--savemodel_period", default=param.savemodel_period, type=int, help="savemodel_period")
#-----------------------------m4_gan_network-----------------------------
'''

# -----------------------------m4_BE_GAN_network-----------------------------

parser.add_argument("--is_train", default=param.is_train, type=int, help="Train")
parser.add_argument("--dataset_dir", default=param.dataset_dir, type=str, help="Train data set dir")
parser.add_argument("--dataset_name", default=param.dataset_name, type=str, help="Train data set name")
parser.add_argument("--datalabel_dir", default=param.datalabel_dir, type=str, help="Train data label dir")
parser.add_argument("--datalabel_name", default=param.datalabel_name, type=str, help="Train data label name")
parser.add_argument("--log_dir", default=param.log_dir, type=str, help="Train data label name")
parser.add_argument("--sampel_save_dir", default=param.sampel_save_dir, type=str, help="sampel save dir")
parser.add_argument("--checkpoint_dir", default=param.checkpoint_dir, type=str, help="model save dir")
parser.add_argument("--num_gpus", default=param.num_gpus, type=int, help="num of gpu")
parser.add_argument("--epoch", default=param.epoch, type=int, help="epoch")
parser.add_argument("--batch_size", default=param.batch_size, type=int, help="batch size for one gpus")
parser.add_argument("--z_dim", default=param.z_dim, type=int, choices=[64, 128], help="dim of noise")
parser.add_argument("--conv_hidden_num", default=param.conv_hidden_num, type=int, choices=[64, 128],
                    help="conv_hidden_num")
parser.add_argument("--data_format", default=param.data_format, type=str, help="data_format")
parser.add_argument("--g_lr", default=param.g_lr, type=float, help="learning rate of G")
parser.add_argument("--d_lr", default=param.d_lr, type=float, help="learning rate of D")
parser.add_argument("--lr_lower_boundary", default=param.lr_lower_boundary, type=float, help="lower learning rate")
parser.add_argument("--gamma", default=param.gamma, type=float, help="gamma")
parser.add_argument("--lambda_k", default=param.lambda_k, type=float, help="lambda_k")
parser.add_argument("--saveimage_period", default=param.saveimage_period, type=int, help="saveimage_period")
parser.add_argument("--savemodel_period", default=param.savemodel_period, type=int, help="savemodel_period")
# -----------------------------m4_BE_GAN_network-----------------------------

# -----------------------------expression,shape,pose-----------------------------
parser.add_argument("--mesh_folder", default=param.mesh_folder, type=str, help="mesh_folder")
parser.add_argument("--train_imgs_mean_file_path", default=param.train_imgs_mean_file_path, type=str,
                    help="Load perturb_Oxford_train_imgs_mean.npz")
parser.add_argument("--train_labels_mean_std_file_path", default=param.train_labels_mean_std_file_path, type=str,
                    help="Load perturb_Oxford_train_labels_mean_std.npz")
parser.add_argument("--ThreeDMM_shape_mean_file_path", default=param.ThreeDMM_shape_mean_file_path, type=str,
                    help="Load 3DMM_shape_mean.npy")
parser.add_argument("--PAM_frontal_ALexNet_file_path", default=param.PAM_frontal_ALexNet_file_path, type=str,
                    help="Load PAM_frontal_ALexNet.npy")
parser.add_argument("--ShapeNet_fc_weights_file_path", default=param.ShapeNet_fc_weights_file_path, type=str,
                    help="Load ShapeNet_fc_weights.npz")
parser.add_argument("--ExpNet_fc_weights_file_path", default=param.ExpNet_fc_weights_file_path, type=str,
                    help="Load ResNet/ExpNet_fc_weights.npz")
parser.add_argument("--fpn_new_model_ckpt_file_path", default=param.fpn_new_model_ckpt_file_path, type=str,
                    help="Load model_0_1.0_1.0_1e-07_1_16000.ckpt")
parser.add_argument("--Shape_Model_file_path", default= param.Shape_Model_file_path, type=str,
                    help="Load ini_ShapeTextureNet_model.ckpt")
parser.add_argument("--Expression_Model_file_path", default=param.Expression_Model_file_path, type=str,
                    help="Load ini_exprNet_model.ckpt")
parser.add_argument("--BaselFaceModel_mod_file_path", default=param.BaselFaceModel_mod_file_path, type=str,
                    help="Load BaselFaceModel_mod.mat")
# -----------------------------expression,shape,pose-----------------------------

cfg = parser.parse_args()

if __name__ == '__main__':
    if not os.path.exists(cfg.log_dir):
        os.makedirs(cfg.log_dir)
    if not os.path.exists(cfg.sampel_save_dir):
        os.makedirs(cfg.sampel_save_dir)
    if not os.path.exists(cfg.mesh_folder):
        os.makedirs(cfg.mesh_folder)
    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(config=config) as sess:
        my_gan = my_gan(sess, cfg)
        if cfg.is_train:
            my_gan.train()
        else:
            print('only train model, please set is_train==True')
            my_gan.ESP_test()