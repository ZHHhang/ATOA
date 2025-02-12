# What is ATOA
Adaptive Trajectory Learning with Obstacle Awareness (ATOA) is a framework for motion planning.
ATOA allows the network to adaptively predict intermediate states, with the potential for more efficient planning solutions. Obstacle information is explicitly integrated by penalizing predictions with obstacle collisions. CDPC module resolves infeasible paths by exploring alternative routes based on direction confidences. 

# What does this repository contain

This repository:
1. Contains the file `train.py` used for training ATOA.  
2. Contains the file `test.py` used for testing ATOA.  
3. Contains the remaining Python files used in conjunction with `train.py` and `test.py`.  
4. Will soon include dataset files for training.

# Dependencies/Instalation
ATOA
1. Pytorch

ATOA can be reliably run on multiple versions of Ubuntu. Testing has confirmed that it can be trained and tested on Ubuntu 18.04–22.04 with their corresponding PyTorch versions.  

However, if you intend to run the environment used for comparative experiments in ATOA (Motionbenchmaker), it is recommended to use Ubuntu 20.04 or later. This is because the MoveIt version in ROS for Ubuntu 18.04 does not include advanced motion planners such as BIT* and AIT*.

# Explanation of files in this repository
1. data_loader.py
2. AE_R_3d_CNN/CNN_3d.py
3. model_R_3D.py
4. neuralplanner.py
5. train_new_orientation_crossentropy_R_3D.py
The main file for training contains all the necessary loss functions and training parameters required for the training process.  
    Before training starts, `train.py` will:  
    - Load data from `data_loader.py`  
    - Load the **environment encoder model** from `AE_R_3d_CNN/CNN_3d.py`  
    - Load the **motion planner model** from `model_R_3D.py`   
# How to Run
1. Assuming paths to demonstration dataset is declared, run the following file to train your ATOA model:
  python ATOA/train.py
2. Assuming paths to demonstration dataset and the trained model are declared, run to test the model:
  python ATOA/test.py

# References
If you use ATOA please consider citing our corresponding article:

Comming soon
