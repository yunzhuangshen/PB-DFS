import os, sys
import argparse
import numpy as np
import scipy.sparse
import scipy.io as sio
from itertools import combinations
from os.path import expanduser
from os import path
import re
from functools import cmp_to_key
import random
import gurobipy as gp
from gurobipy import *


class Graph:
    """
    Container for a graph.
    Parameters
    ----------
    number_of_nodes : int
        The number of nodes in the graph.
    edges : set of tuples (int, int)
        The edges of the graph, where the integers refer to the nodes.
    degrees : numpy array of integers
        The degrees of the nodes in the graph.
    neighbors : dictionary of type {int: set of ints}
        The neighbors of each node in the graph.
    """
    def __init__(self, number_of_nodes, edges, degrees, neighbors):
        self.number_of_nodes = number_of_nodes
        self.edges = edges
        self.degrees = degrees
        self.neighbors = neighbors

    def __len__(self):
        """
        The number of nodes in the graph.
        """
        return self.number_of_nodes

    @staticmethod
    def barabasi_albert(number_of_nodes, affinity, random):
        """
        Generate a Barabási-Albert random graph with a given edge probability.
        Parameters
        ----------
        number_of_nodes : int
            The number of nodes in the graph.
        affinity : integer >= 1
            The number of nodes each new node will be attached to, in the sampling scheme.
        random : numpy.random.RandomState
            A random number generator.
        Returns
        -------
        Graph
            The generated graph.
        """
        assert affinity >= 1 and affinity < number_of_nodes

        edges = set()
        degrees = np.zeros(number_of_nodes, dtype=int)
        neighbors = {node: set() for node in range(number_of_nodes)}
        for new_node in range(affinity, number_of_nodes):
            # first node is connected to all previous ones (star-shape)
            if new_node == affinity:
                neighborhood = np.arange(new_node)
            # remaining nodes are picked stochastically
            else:
                neighbor_prob = degrees[:new_node] / (2*len(edges))
                neighborhood = random.choice(new_node, affinity, replace=False, p=neighbor_prob)
            for node in neighborhood:
                edges.add((node, new_node))
                degrees[node] += 1
                degrees[new_node] += 1
                neighbors[node].add(new_node)
                neighbors[new_node].add(node)

        graph = Graph(number_of_nodes, edges, degrees, neighbors)
        return graph


def generate_ds(graph, filename):
    """
    Generate a Maximum Independent Set (also known as Maximum Stable Set) instance
    in CPLEX LP format from a previously generated graph.
    Parameters
    ----------
    graph : Graph
        The graph from which to build the independent set problem.
    filename : str
        Path to the file to save.
    """


    with open(filename, 'w') as lp_file:
        lp_file.write("minimize\nOBJ:" + "".join([f" + 1 x{node+1}" for node in range(len(graph))]) + "\n")
        lp_file.write("\nsubject to\n")
        for count, node in enumerate(range(len(graph))):
            neighbors = graph.neighbors[node]
            cons = f"+1 x{node+1}" + "".join([f" +1 x{j+1}" for j in neighbors])
            lp_file.write(f"C{count+1}: {cons} >= 1\n")
        lp_file.write("\nbinary\n" + " ".join([f"x{node+1}" for node in range(len(graph))]) + "\n")


def solve_single(lp_path, sol_path, time_limit=500):
    print(f'process lp: {lp_path}')

    model = gp.read(lp_path)
    model.setParam('TimeLimit', time_limit) # set a time limit
    model.setParam('OutputFlag', 0) # disable logging
    model.optimize()
    if model.status != GRB.OPTIMAL:
        print(f'problem is too hard to solve within {time_limit}, skipping!')
        return
    
    print(f'problem is solved with {round(model.runtime, 1)} seconds')
    with open( sol_path, 'w+') as f:
        # f.write('Obj: %f\n' % model.objVal)
        for v in model.getVars():
            if int(v.x) == 1 and v.varName[0] == 'x' :
                f.write(f'{v.varName[1:]}\n')


def solve_ds(lp_path, time_limit=200):
    sol_file_path =  f'{lp_path[:-2]}sol'
    if os.path.exists(sol_file_path):
        print(f'{lp_path} has been processed, skipping!')
        return

    solve_single(lp_path, sol_file_path, time_limit=time_limit)
    sys.stdout.flush()


def gen_ds(data_dir, ninst, scale_lower, scale_upper=None, solve=True):
    
    os.makedirs(data_dir, exist_ok=True)
    affinity = 4

    for i in range(ninst):
        nnodes = scale_lower if scale_upper is None else random.randint(scale_lower, scale_upper+1)
        graph = Graph.barabasi_albert(nnodes, affinity, np.random.RandomState(i))

        lp_path = os.path.join(data_dir, f'{i}.lp')
        generate_ds(graph, lp_path)
        
        if solve:
            sol_path = solve_ds(lp_path)

if __name__ == '__main__':

    home = expanduser("~")
    # data_dir = os.path.join(home, f'storage1/instances/ds/train_500-1000')
    # gen_ds(data_dir, 500, 500, 1000, solve=True)

    # data_dir = os.path.join(home, f'storage1/instances/ds/time_1000')
    # gen_ds(data_dir, 30, 1000, solve=False)

    # data_dir = os.path.join(home, f'storage1/instances/ds/time_3000')
    # gen_ds(data_dir, 30, 3000, solve=False)

    # data_dir = os.path.join(home, f'storage1/instances/ds/time_5000')
    # gen_ds(data_dir, 30, 5000, solve=False)

    # data_dir = os.path.join(home, f'storage1/instances/ds/time_7000')
    # gen_ds(data_dir, 30, 7000, solve=False)

    # data_dir = os.path.join(home, f'storage1/instances/ds/time_9000')
    # gen_ds(data_dir, 30, 9000, solve=False)
