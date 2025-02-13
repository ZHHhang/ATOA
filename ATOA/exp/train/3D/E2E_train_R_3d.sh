cd ../../../ 
python train_3D.py --model_path ./models/crossentropy/test/CNN_NEWLOSS_3D/NP_400_dropout_0.2  \
--batch_size_train 20 --dropout_p_MLP 0.2 \
--load_data_N 10 --load_data_NP 40 --s 50 --sp 0 \
--save_epoch 2 --test_epochs 2 \
--T_data_N 10 --T_data_NP 40 --T_data_s 50 --T_data_sp 4000 \
--UT_data_N 10 --UT_data_NP 40 --UT_data_s 100 --UT_data_sp 0 \
--seen_environment_test 1 --Unseen_environment_test 1 \
--scale_para 3.2 --bias 20.0 \
--interpolation_point_num 10 --dropout_p_CNN 0.0 \
--batch_size_test 20 \
--k1 1 --k2_theta 4.0 --k2_phi 4.0 --k3 0.1 --k4 2 \
--ek 2 \
--input_size 66 
cd exp

# --num_epochs 4000 --preload 0 \
# --num_sector 180 --batch_size_train 100 --learning_rate 0.0001 \
# --interpolation_point_num 50 --dropout_p 0.0 \
# --load_data_N 20 --load_data_NP 400 \
# --k1 1 --k2 4.0 --k3 0.1 --k4 2 \
# --ek 2 \
# --save_epoch 50