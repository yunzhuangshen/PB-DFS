#! /bin/bash

nohup python3 TRIG-GCN/train.py mis &
nohup python3 TRIG-GCN/train.py vc &
nohup python3 TRIG-GCN/train.py ds &
nohup python3 TRIG-GCN/train.py ca &