cd ../../
python train.py --model_path ./models/crossentropy/test/CNN_NEWLOSS/dropout_0.2  \
--scale_para 4 --bias 20.0 --num_epochs 4000 --preload_all 0 --preload_encoder 1 \
--num_sector 180 --batch_size_train 100 \
--learning_rate_together 0.0001 --learning_rate_encoder 0.00001 --learning_rate_planner 0.0001 \
--interpolation_point_num 50 --dropout_p_MLP 0.2 --dropout_p_CNN 0.0 \
--load_data_N 30 --load_data_NP 200 \
--T_data_N 10 --T_data_NP 40 --T_data_s 90 --T_data_sp 4000 \
--UT_data_N 10 --UT_data_NP 40 --UT_data_s 100 --UT_data_sp 0 \
--k1 1 --k2 4.0 --k3 0.1 --k4 2 \
--Tk1 1 --Tk2 4.0 --Tk3 0.1 --Tk4 2 \
--ek 2 \
--save_epoch 10 --test_epochs 10 \
--batch_size_test 100 --seen_environment_test 1 --Unseen_environment_test 1 \

cd exp

