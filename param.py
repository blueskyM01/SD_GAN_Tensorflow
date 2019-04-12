import tensorflow as tf

# -----------------------------m4_BE_GAN_network-----------------------------
gpu_assign = '0'
is_train = True
save_dir = '/muti-progress-train_tttt/'
dataset_dir = '/media/yang/F/DataSet/Face'
dataset_name = 'ms1s_align'
datalabel_dir = '/media/yang/F/DataSet/Face/Label'
datalabel_name = 'MS-Celeb-1M_clean_list.txt'
face_model_dir = '/media/yang/F/ubuntu/SD_GAN_Result/face_model_ms1s_continue_1/checkpoint'
face_model_name = 'ms1s_align'
tfrecord_path = '/media/yang/F/DataSet/Face/ms1s_tfrecords_1'
BE_GAN_model_dir = '/home/yang/study/datasetandparam/param_gan'
BE_GAN_model_name = 'Pre_Gan1'
experiment_result_dir = '/media/yang/F/ubuntu/SD_GAN_Result'
log_dir = experiment_result_dir + save_dir + 'logs'  # need to change
sampel_save_dir = experiment_result_dir + save_dir + 'samples'  # need to change
checkpoint_dir = experiment_result_dir + save_dir + 'checkpoint'  # need to change
test_sample_save_dir = experiment_result_dir + save_dir + 'test_sample2'  # need to change
num_gpus = 2
epoch = 10
batch_size = 8  # need to change
z_dim = 128  # or 128
conv_hidden_num = 128  # 128
data_format = 'NHWC'
g_lr = 0.00008  # need to change
d_lr = 0.00008  # need to change
lr_lower_boundary = 0.00002
gamma = 0.5
lambda_k = 0.01
add_summary_period = 100
saveimage_period = 1
saveimage_idx = 500
savemodel_period = 1
lr_drop_period = 1
lambda_s = 0.05
lambda_e = 0.05
lambda_p = 0.1
lambda_id = 1.0

# -----------------------------m4_BE_GAN_network-----------------------------

mesh_folder = 'output_ply'
ThreeDMM_Param_dir = '/media/yang/F/DataSet/Face/param_of_SD_GAN/ThreeDMM_Param'
train_imgs_mean_file_path = ThreeDMM_Param_dir + '/fpn_new_model/perturb_Oxford_train_imgs_mean.npz'
train_labels_mean_std_file_path = ThreeDMM_Param_dir + '/fpn_new_model/perturb_Oxford_train_labels_mean_std.npz'
ThreeDMM_shape_mean_file_path = ThreeDMM_Param_dir + '/Shape_Model/3DMM_shape_mean.npy'
PAM_frontal_ALexNet_file_path = ThreeDMM_Param_dir + '/fpn_new_model/PAM_frontal_ALexNet.npy'
ShapeNet_fc_weights_file_path = ThreeDMM_Param_dir + '/ShapeNet_fc_weights.npz'
ExpNet_fc_weights_file_path = ThreeDMM_Param_dir + '/ExpNet_fc_weights.npz'
fpn_new_model_ckpt_file_path = ThreeDMM_Param_dir + '/fpn_new_model/model_0_1.0_1.0_1e-07_1_16000.ckpt'
Shape_Model_file_path = ThreeDMM_Param_dir + '/Shape_Model/ini_ShapeTextureNet_model.ckpt'
Expression_Model_file_path = ThreeDMM_Param_dir + '/Expression_Model/ini_exprNet_model.ckpt'
BaselFaceModel_mod_file_path = ThreeDMM_Param_dir + '/Shape_Model/BaselFaceModel_mod.mat'
# -----------------------------expression,shape,pose-----------------------------
