cd ../../ 
python neuralplanner_CrossEntropy_R3D_arb_new_loss_e2e.py \
--MLP_path "./models/crossentropy/test/CNN_NEWLOSS_3D/test_use/MLP_E2E_mask_3d_epoch=200_N_10_Np_1800_DropMLP=0.20.pkl" \
--CNN_path "./models/crossentropy/test/CNN_NEWLOSS_3D/test_use/CNN_E2E_mask_3d_epoch=200_N_10_Np_1800_DropMLP=0.20.pkl" \
--input_size 66 --num_sector_theta 90 --num_sector_phi 180 \
--enviro_index_low 30 --enviro_index_up 34 \
--Path_index_low 0 --Path_index_up 1000 \
--Plot_flag 1 --Path_lvc_state 0 \
--stuck_search_time 1000 --Sparse_index 5
cd exp

#--scale_para 4 --bias 20.0  need to be control on top of the script
# --num_sector 180 --dropout_p 0.0 --Plot_flag 1 \


# --Path_lvc_state 0
