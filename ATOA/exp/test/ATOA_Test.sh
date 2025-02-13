cd ../../ 
python test_3D.py \
--input_size 66 --num_sector_theta 90 --num_sector_phi 180 \
--enviro_index_low 0 --enviro_index_up 2 \
--Path_index_low 2 --Path_index_up 4 \
--Plot_flag 1 --Path_lvc_state 1 \
--stuck_search_time 90 --Sparse_index 5
cd exp

#--scale_para 4 --bias 20.0  need to be control on top of the script
# --num_sector 180 --dropout_p 0.0 --Plot_flag 1 \


# --Path_lvc_state 0