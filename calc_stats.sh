#! /bin/bash

python3 stats.py mis > ret_solver/mis.txt
python3 stats.py vc > ret_solver/vc.txt
python3 stats.py ca > ret_solver/ca.txt
python3 stats.py ds > ret_solver/ds.txt
