# ATOA
Adaptive Trajectory Learning with Obstacle Awareness (ATOA) is a framework for motion planning.
ATOA allows the network to adaptively predict intermediate states, with the potential for more efficient planning solutions. 
This repository:
1. Contains the file `train.py` used for training ATOA.  
2. Contains the file `test.py` used for testing ATOA.  
3. Contains the remaining Python files used in conjunction with `train.py` and `test.py`.  
4. Will soon include dataset files for training.

# Requirements
ATOA
1. Pytorch

ATOA can be reliably run on multiple versions of Ubuntu. Testing has confirmed that it can be trained and tested on Ubuntu 18.04–22.04 with their corresponding PyTorch versions.  

However, if you intend to run the environment used for comparative experiments in ATOA (Motionbenchmaker), it is recommended to use Ubuntu 20.04 or later. This is because the MoveIt version in ROS for Ubuntu 18.04 does not include advanced motion planners such as BIT* and AIT*.

# How to Run
1. Assuming paths to demonstration dataset is declared, run the following file to train your ATOA model:
  python ATOA/train.py
2. Assuming paths to demonstration dataset and the trained model are declared, run to test the model:
  python ATOA/test.py
