python run_this.py \
--gpu_assign='1' \
--is_train=True \
--log_dir='/media/yang/F/ubuntu/SD_GAN_Result/with pose shape expression lambdak=0.001/logs' \
--sampel_save_dir='/media/yang/F/ubuntu/SD_GAN_Result/with pose shape expression lambdak=0.001/samples' \
--checkpoint_dir='/media/yang/F/ubuntu/SD_GAN_Result/with pose shape expression lambdak=0.001/checkpoint' \
--test_sample_save_dir='/media/yang/F/ubuntu/SD_GAN_Result/with pose shape expression lambdak=0.001/test_sample' \
--lambda_k=0.001 \
--lambda_s=0.05 \
--lambda_e=0.05 \
--lambda_p=0.1 \
--lambda_id=1.0 \
--add_summary_period=100 \
--lr_drop_period=1