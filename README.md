# Learning Primal Heuristics for Mixed Integer Programs

## Requirements

#### Python Code Dependencies 
1. We use python version 3.6.9.
2. Install python3-venv by 'sudo apt-get install python3-venv'
3. We need to create two virtual environments that contain different version of tensorflow, namely tf1 and tf2.
4. Make sure the cmake version >= 3.15
5. Make sure pip is upgraded to the latest version under both environments, 'pip3 install -U pip' 
5. Under the environment tf1, run 'pip3 install -r requirements_1.txt'
6. Under the environment tf2, run 'pip3 install -r requirements_2.txt'

#### C++ Code Dependencies
1. C++ boost library is required. The library can be downloaded here https://www.boost.org/users/download/.
2. SCIP solver **version 6.0.1** is required, which can be downloaded at https://www.scipopt.org/index.php#download. An academic license can be applied from https://www.scipopt.org/index.php#license.
3. After the setup, build the c++ code with cmake to obtain the execuable 'CO'.


#### Data and Pretrained Models
Please request access to the full codebase at https://drive.google.com/file/d/1y-y_sijEoR8eYVFJsUpM9FXIgLNajCCQ/view?usp=sharing.

## Model Training

- To train GG-GCN, XGBoost, and LR models, activate the tf1 environment and then run the bash script ./model_train.sh.

- To train TRIG-GCN model, active the tf2 environment and then run the bash script ./model_train_trig_gcn.sh

## Model Testing

- To test GG-GCN, XGBoost, and LR models, activate the tf1 environment and then run the bash script './model_test.sh'.

- To test TRIG-GCN model, active the tf2 environment and then hen run the bash script './model_test_trig_gcn.sh'.

The testing results is output to the folder 'ret_model'. These results corresponds to the results presented in Table 1 of our paper.

## Evaluation of Heuristics

- Run the bash script './model_predict.sh' to produce solution predictions for the proposed heuristic. Or skip this step and use the provided solution predictions in the dataset.

- Run the bash script ./heur_eval.sh. It takes several hours to obtain the results. Please note that if each process should run on a single cpu. The intermediate results is output to the folder ret_solver.

- Upon the previous step is finished, run the bash script './calc_stats.sh' (under tf1 environment) to generate the mean statistics, which is output to folder 'ret_solver'. These results correspond to the statistics in Table 2 and Table 3 of our paper. 