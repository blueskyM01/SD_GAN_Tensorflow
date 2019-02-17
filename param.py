import tensorflow as tf

'''
#-----------------------------m4_gan_network-----------------------------
dataset_dir = '/media/yang/F/DataSet/Face'
dataset_name = 'lfw-deepfunneled'
datalabel_dir = '/media/yang/F/DataSet/Face/Label'
datalabel_name = 'pair_FGLFW.txt'
log_dir = './logs'
sampel_save_dir = './samples'
num_gpus = 2
epoch = 40
learning_rate = 0.001
beta1 = 0.5
beta2 = 0.5
batch_size = 16
z_dim = 128
g_feats = 64
saveimage_period = 10
savemodel_period = 40
#-----------------------------m4_gan_network-----------------------------
'''

# -----------------------------m4_BE_GAN_network-----------------------------
is_train = True
save_dir = '/CASIA-WebFace_align_train/'
dataset_dir = '/media/yang/F/DataSet/Face'
dataset_name = 'CASIA-WebFace_align'
datalabel_dir = '/media/yang/F/DataSet/Face/Label'
datalabel_name = 'CASIA-WebFace_align.txt'
log_dir = '/media/yang/F/ubuntu/SD_GAN_Result' + save_dir+'logs'  # need to change
sampel_save_dir = '/media/yang/F/ubuntu/SD_GAN_Result' + save_dir+'samples'  # need to change
checkpoint_dir = '/media/yang/F/ubuntu/SD_GAN_Result' + save_dir+'checkpoint'  # need to change
test_sample_save_dir = '/media/yang/F/ubuntu/SD_GAN_Result' + save_dir+'test_sample'  # need to change
num_gpus = 1
epoch = 100
batch_size = 16  # need to change
z_dim = 64
conv_hidden_num = 128 # 128
data_format = 'NHWC'
g_lr = 0.00008  # need to change
d_lr = 0.00008  # need to change
lr_lower_boundary = 0.00001
gamma = 0.5
lambda_k = 0.001
add_summary_period = 50
saveimage_period = 1
savemodel_period = 10
lr_drop_period = 30
# -----------------------------m4_BE_GAN_network-----------------------------

mesh_folder = 'output_ply'
train_imgs_mean_file_path = '/home/yang/My_Job/fpn_new_model/perturb_Oxford_train_imgs_mean.npz'
train_labels_mean_std_file_path = '/home/yang/My_Job/fpn_new_model/perturb_Oxford_train_labels_mean_std.npz'
ThreeDMM_shape_mean_file_path = '/home/yang/My_Job/Shape_Model/3DMM_shape_mean.npy'
PAM_frontal_ALexNet_file_path = '/home/yang/My_Job/fpn_new_model/PAM_frontal_ALexNet.npy'
ShapeNet_fc_weights_file_path = '/home/yang/My_Job/study/Expression-Net/ResNet/ShapeNet_fc_weights.npz'
ExpNet_fc_weights_file_path = '/home/yang/My_Job/study/Expression-Net/ResNet/ExpNet_fc_weights.npz'
fpn_new_model_ckpt_file_path = '/home/yang/My_Job/fpn_new_model/model_0_1.0_1.0_1e-07_1_16000.ckpt'
Shape_Model_file_path = '/home/yang/My_Job/Shape_Model/ini_ShapeTextureNet_model.ckpt'
Expression_Model_file_path = '/home/yang/My_Job/Expression_Model/ini_exprNet_model.ckpt'
BaselFaceModel_mod_file_path = '/home/yang/My_Job/Shape_Model/BaselFaceModel_mod.mat'
# -----------------------------expression,shape,pose-----------------------------
