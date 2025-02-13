cd ../../../ 
python test_2D.py \
--num_sector 180 --dropout_p 0.0 --Plot_flag 1 \
--enviro_index_low 5 --enviro_index_up 6 \
--Path_index_low 5 --Path_index_up 6 \
--Path_lvc_state 1
cd exp

#--scale_para 4 --bias 20.0  need to be control on top of the script
