#! /bin/bash

nohup python3 GG-GCN/pred_gcn.py mis &
nohup python3 GG-GCN/pred_gcn.py vc &
nohup python3 GG-GCN/pred_gcn.py ds &
nohup python3 GG-GCN/pred_gcn.py ca &

nohup python3 GG-GCN/pred_baselines.py mis -m lr &
nohup python3 GG-GCN/pred_baselines.py vc -m lr &
nohup python3 GG-GCN/pred_baselines.py ds -m lr &
nohup python3 GG-GCN/pred_baselines.py ca -m lr &