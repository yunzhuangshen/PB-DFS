#! /bin/bash
prefix=build

# Combinatorial Auction Problem
nohup ${prefix}/CO -p 6 -h 0 &
nohup ${prefix}/CO -p 6 -h 2 &
nohup ${prefix}/CO -p 6 -h 4 &

nohup ${prefix}/CO -p 6 -h 4 -t 50 &
nohup ${prefix}/CO -p 6 -h 6 -t 50 &
nohup ${prefix}/CO -p 6 -h 7 -t 50 &
nohup ${prefix}/CO -p 6 -h 8 -t 50 &
nohup ${prefix}/CO -p 6 -h 9 -t 50 &
nohup ${prefix}/CO -p 6 -h 10 -t 50 &

# Dominant Set Problem
nohup ${prefix}/CO -p 5 -h 0 &
nohup ${prefix}/CO -p 5 -h 2 &
nohup ${prefix}/CO -p 5 -h 3 &

nohup ${prefix}/CO -p 5 -h 3 -t 50 &
nohup ${prefix}/CO -p 5 -h 5 -t 50 &   
nohup ${prefix}/CO -p 5 -h 7 -t 50 &
nohup ${prefix}/CO -p 5 -h 8 -t 50 &
nohup ${prefix}/CO -p 5 -h 9 -t 50 &
nohup ${prefix}/CO -p 5 -h 10 -t 50 &

# # Vertex Cover Problem
nohup ${prefix}/CO -p 4 -h 0 &
nohup ${prefix}/CO -p 4 -h 2 &
nohup ${prefix}/CO -p 4 -h 4 &

nohup ${prefix}/CO -p 4 -h 4 -t 50 &
nohup ${prefix}/CO -p 4 -h 6 -t 50 &
nohup ${prefix}/CO -p 4 -h 7 -t 50 &
nohup ${prefix}/CO -p 4 -h 8 -t 50 &
nohup ${prefix}/CO -p 4 -h 9 -t 50 &
nohup ${prefix}/CO -p 4 -h 10 -t 50 &

# # Maximum Independent Set Problem
nohup ${prefix}/CO -p 0 -h 0 &
nohup ${prefix}/CO -p 0 -h 2 &
nohup ${prefix}/CO -p 0 -h 4 &

nohup ${prefix}/CO -p 0 -h 4 -t 50 &
nohup ${prefix}/CO -p 0 -h 6 -t 50 &
nohup ${prefix}/CO -p 0 -h 7 -t 50 &
nohup ${prefix}/CO -p 0 -h 8 -t 50 &
nohup ${prefix}/CO -p 0 -h 9 -t 50 &
nohup ${prefix}/CO -p 0 -h 10 -t 50 &
