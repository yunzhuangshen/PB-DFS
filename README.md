# Learning Primal Heuristics for Mixed Integer Programs

## Requirements

#### Python Code Dependencies 
1. Python version 3.6.9.
2. Cuda version 10.0 (required by TRIG-GNN)
3. Cmake version >= 3.15
4. python3-venv (installed by running 'sudo apt-get install python3-venv')
5. Two virtual environments that contain different version of tensorflow, tf1 and tf2. (created by running 'python3 -m venv [env_name]').
6. Latest pip (upgraded by running 'pip3 install -U pip' 
7. Install dependencies for the two environments:
    - Activate the environment tf1 (source tf1/bin/activate), run 'pip3 install -r requirements_1.txt'
    - Activate the environment tf2 (source tf2/bin/activate), run 'pip3 install -r requirements_2.txt'

#### C++ Code Dependencies
1. C++ boost library is required. The library can be downloaded here https://www.boost.org/users/download/.
2. SCIP solver **version 6.0.1** is required, which can be downloaded at https://www.scipopt.org/index.php#download. An academic license can be applied from https://www.scipopt.org/index.php#license.
3. After the setup, build the c++ code with cmake to obtain the execuable 'CO'.


#### Datasets
Datasets are available at https://drive.google.com/file/d/1HBBdwtQ1fa31inb9wVNcT-Tslu4WAeny/view?usp=sharing.  


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

## Generating your own training/test problem instances
- If you need to generate your own training and test instances, you can use the code in "data_generator" directory. Each problem directory contains two files:
- gen_inst_*: generate problem instances with different parameters and/or solve the problem instances to optimality.
- make_sample_*: extract features for problem instances and make training data.

Two python packages are required for data generation:
- gurobipy (for solving training instances to optimality).
- PySCIPOpt (for feature extraction). Note that to have the feature extraction code please install our version of PySCIPOpt included in this project.

## FAQ
- Currently, the ML models are implemented in the python code, and their predictions are wrriten into the filesystem to be used by the scip implemented in C++.
- If you need to predict and search interactively, you may want to have a look at "PySCIPOpt", which binds python and C++ using cython.    
- If you have any questions, please contact me at shenyunzhuang@outlook.com. Hopefully, the code can be helpful for your own reasearch. 