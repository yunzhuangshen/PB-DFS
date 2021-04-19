#! /bin/bash

nohup python3 GG-GCN/test_gcn.py mis &
nohup python3 GG-GCN/test_gcn.py vc &
nohup python3 GG-GCN/test_gcn.py ds &
nohup python3 GG-GCN/test_gcn.py ca &

nohup python3 GG-GCN/test_baselines.py mis -m lr &
nohup python3 GG-GCN/test_baselines.py vc -m lr &
nohup python3 GG-GCN/test_baselines.py ds -m lr &
nohup python3 GG-GCN/test_baselines.py ca -m lr &

nohup python3 GG-GCN/test_baselines.py mis -m xgb &
nohup python3 GG-GCN/test_baselines.py vc -m xgb &
nohup python3 GG-GCN/test_baselines.py ds -m xgb &
nohup python3 GG-GCN/test_baselines.py ca -m xgb &