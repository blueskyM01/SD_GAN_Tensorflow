from __future__ import division, print_function, absolute_import
import os
import argparse
import tensorflow as tf
import param
from model import my_gan
import time

os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定第  块GPU可用

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
# TF_CPP_MIN_LOG_LEVEL 取值 0 ： 0也是默认值，输出所有信息
# TF_CPP_MIN_LOG_LEVEL 取值 1 ： 屏蔽通知信息
# TF_CPP_MIN_LOG_LEVEL 取值 2 ： 屏蔽通知信息和警告信息
# TF_CPP_MIN_LOG_LEVEL 取值 3 ： 屏蔽通知信息、警告信息和报错信息


parser = argparse.ArgumentParser()

# -----------------------------m4_BE_GAN_network-----------------------------

parser.add_argument("--is_train", default=param.is_train, type=int, help="Train")
parser.add_argument("--dataset_dir", default=param.dataset_dir, type=str, help="Train data set dir")
parser.add_argument("--dataset_name", default=param.dataset_name, type=str, help="Train data set name")
parser.add_argument("--datalabel_dir", default=param.datalabel_dir, type=str, help="Train data label dir")
parser.add_argument("--datalabel_name", default=param.datalabel_name, type=str, help="Train data label name")
parser.add_argument("--log_dir", default=param.log_dir, type=str, help="Train data label name")
parser.add_argument("--sampel_save_dir", default=param.sampel_save_dir, type=str, help="sampel save dir")
parser.add_argument("--checkpoint_dir", default=param.checkpoint_dir, type=str, help="model save dir")
parser.add_argument("--test_sample_save_dir", default=param.test_sample_save_dir, type=str, help="test sample save dir")
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
parser.add_argument("--add_summary_period", default=param.add_summary_period, type=int, help="add_summary_period")
parser.add_argument("--lr_drop_period", default=param.lr_drop_period, type=int, help="lr_drop_period")
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
parser.add_argument("--Shape_Model_file_path", default=param.Shape_Model_file_path, type=str,
                    help="Load ini_ShapeTextureNet_model.ckpt")
parser.add_argument("--Expression_Model_file_path", default=param.Expression_Model_file_path, type=str,
                    help="Load ini_exprNet_model.ckpt")
parser.add_argument("--BaselFaceModel_mod_file_path", default=param.BaselFaceModel_mod_file_path, type=str,
                    help="Load BaselFaceModel_mod.mat")
# -----------------------------expression,shape,pose-----------------------------

cfg = parser.parse_args()

if __name__ == '__main__':

    config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(config=config) as sess:
        if not os.path.exists(cfg.log_dir):
            os.makedirs(cfg.log_dir)
        if not os.path.exists(cfg.sampel_save_dir):
            os.makedirs(cfg.sampel_save_dir)
        if not os.path.exists(cfg.mesh_folder):
            os.makedirs(cfg.mesh_folder)
        my_gan = my_gan(sess, cfg)
        if cfg.is_train:
            my_gan.train()
        else:
            # print('only train model, please set is_train==True')
            if not os.path.exists():
                os.makedirs(cfg.mesh_folder)
            my_gan.test()
