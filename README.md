# Learning Primal Heuristics for Mixed Integer Programs

This repository contains the experiment code for the paper submission: Learning Primal Heuristics for Mixed Integer Programs


## Requirements

#### Python Code Dependencies 
1. We use python version 3.6.9.
2. Install python3-venv by 'sudo apt-get install python3-venv'
3. We need to create two virtual environments that contain different version of tensorflow, namely tf1 and tf2.
4. Under the environment tf1, run 'pip3 install -r requirements_1.txt'
5. Under the environment tf2, run 'pip3 install -r requirements_2.txt'

#### C++ Code Dependencies
1. C++ boost library is required. The library can be downloaded here https://www.boost.org/users/download/.
2. SCIP solver **version 6.0.1** is required, which can be downloaded at https://www.scipopt.org/index.php#download. An academic license can be applied from https://www.scipopt.org/index.php#license.
3. After the setup, build the c++ code with cmake by 'cd build && cmake ../ && make -j4' to obtain the execuable 'CO'.


#### Data Dependencies
The data is available at . The datasets repository should be put under the root directory of the project.

## Model Training

- To train GG-GCN, XGBoost, and LR models, we first activate the tf1 environment. Then run the bash script ./model_train.sh.

- To train TRIG-GCN model, we active the tf2 environment. Then run the bash script ./model_train_trig_gcn.sh

## Pre-trained Models

We provide the pre-trained models for each problem under the folder 'trained_models'

## Model Testing

- To test GG-GCN, XGBoost, and LR models, we first activate the tf1 environment. Then run the bash script './model_test.sh'.

- To test TRIG-GCN model, we active the tf2 environment. Then run the bash script './model_test_trig_gcn.sh'.

The testing results is output to the folder 'ret_model'. These results corresponds to the results presented in Table 1 of our paper.

## Evaluation of Heuristics

- Run the bash script ./heur_eval.sh. It takes several hours to obtain the results.

- Upon the previous step is finished, run the python script 'stats.py' to generate the mean statistics, which is output to folder 'ret_solver'. These results correspond to the statistics in Table 2 and Table 3 of our paper. 