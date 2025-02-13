# What is ATOA
Adaptive Trajectory Learning with Obstacle Awareness (ATOA) is a framework for motion planning.
ATOA allows the network to adaptively predict intermediate states, with the potential for more efficient planning solutions. Obstacle information is explicitly integrated by penalizing predictions with obstacle collisions. CDPC module resolves infeasible paths by exploring alternative routes based on direction confidences. 

# What does the ATOA folder contain 

1. Contains the file `train.py` used for training ATOA.  
2. Contains the file `test.py` used for testing ATOA.
3. Contains the file exp/train/ATOA_train.sh, which is used to quickly complete the parameter configuration for the ATOA training file train.py and to launch the train.py.
4. Contains the file exp/train/ATOA_test.sh, which is used to quickly complete the parameter configuration for the ATOA test file test.py and to launch the test.py.
5. Contains the remaining Python files used in conjunction with `train.py` and `test.py`.  
6. Will soon include dataset files for training.

# Dependencies/Instalation
Ubuntu+Cuda+Pytorch

ATOA can be reliably run on multiple versions of Ubuntu. Testing has confirmed that it can be trained and tested on Ubuntu 18.04–22.04 with their corresponding PyTorch versions.  

However, if you intend to run the environment used for comparative experiments in ATOA (Motionbenchmaker), it is recommended to use Ubuntu 20.04 or later. This is because the MoveIt version in ROS for Ubuntu 18.04 does not include advanced motion planners such as BIT* and AIT*.

# Explanation of files in this repository

1. data_loader.py
    Used to specify the path and method for loading the training/testing dataset.
   
2. AE_R_3d_CNN/CNN_3d.py
    This file contains a model with an 3D-environment encoder.

3. AE_concave_2d_CNN/CNN_2d.py
    This file contains a model with an 2D-environment encoder.
   
4. model.py
    This file contains a model of the motion planner.
   
6. train.py
The main file for training contains all the necessary loss functions and training parameters required for the training process.  
    Before training starts, `train.py` will:  
    - Load data from `data_loader.py`  
    - Load the environment encoder model from `AE_R_3d_CNN/CNN_3d.py` 
    - Load the motion planner model from `model.py`
      
7. neuralplanner.py
The main file for testing the trained ATOA model contains all the necessary modules required for the testing process. The CDPC replanning module is also included in this file.  
    Before testing starts, `test.py` will:  
    - Load the test data from `data_loader.py`  
    - Load the environment encoder model from `AE_R_3d_CNN/CNN_3d.py`  
    - Load the motion planner model from `model.py`    
    Note:  
    The CDPC module has not yet been encapsulated as an interface function in `test.py` and currently exists as separate function calls. We will improve this and upload a more readable version of the code.
    
# How to Run

1. Train and test using the .sh script files (recommended).

    (1)  Assuming the path to the demonstration dataset has been declared, run the following script to train your ATOA model and specify the required important parameters:

        bash ATOA/exp/train/ATOA_train.sh
   
    (2)  Assuming the path to the demonstration dataset and the trained model have been declared, run the following script to test your ATOA model and specify the required important parameters:

        bash ATOA/exp/test/ATOA_test.sh
   
3. Train using the original Python file.

    (1). Assuming paths to demonstration dataset is declared, run the following file to train your ATOA model:
      python ATOA/train.py
    Here’s the translation of your provided content:
    
    ---
    
        Before training, the following parameters may need to be modified according to user requirements:
        
        In `ATOA_train.py`:
           (Optional)
           (1) Modify the storage location of the result file:
           ```
           parser.add_argument('--model_path', type=str, default='./Result', help='path for saving trained models')
           ```
           (2) Modify the training set parameters, the default values are N=35, NP=200:
           ```
           parser.add_argument('--load_data_N', type=int, default=35)
           parser.add_argument('--load_data_NP', type=int, default=200)
           ```
           (3) Modify the dropout parameter size:
           ```
           parser.add_argument('--dropout_p', type=float , default=0.5, help='The probability of dropout')
           ```
           (4) Modify the cost function `ek` parameter:
           ```
           parser.add_argument('--ek', type=float, default=4, help='parameter of loss_function')
           ```
           (5) Modify the saved file name:
           ```
           model_path='new_mlp_cross_mask_concave2d_epoch=%d_N_%d_Np_%d_Dropout=%.3f_k3=%.3f_ek=%.3f'
           ```
           (6) Modify the `sm` parameter, i.e., how many epochs before saving the model again.
        
         In `ATOA_data_loader_2d.py`, you can specify all file save and load paths.
    
    ---
    
    
    
    (2). Assuming paths to demonstration dataset and the trained model are declared, run to test the model:
      python ATOA/test.py

# References
If you use ATOA please consider citing our corresponding article:

Comming soon
