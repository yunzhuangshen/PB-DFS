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

    def greedy_clique_partition(self):
        """
        Partition the graph into cliques using a greedy algorithm.
        Returns
        -------
        list of sets
            The resulting clique partition.
        """
        cliques = []
        leftover_nodes = (-self.degrees).argsort().tolist()

        while leftover_nodes:
            clique_center, leftover_nodes = leftover_nodes[0], leftover_nodes[1:]
            clique = {clique_center}
            neighbors = self.neighbors[clique_center].intersection(leftover_nodes)
            densest_neighbors = sorted(neighbors, key=lambda x: -self.degrees[x])
            for neighbor in densest_neighbors:
                # Can you add it to the clique, and maintain cliqueness?
                if all([neighbor in self.neighbors[clique_node] for clique_node in clique]):
                    clique.add(neighbor)
            cliques.append(clique)
            leftover_nodes = [node for node in leftover_nodes if node not in clique]

        return cliques


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


def generate_indset(graph, filename):
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
    cliques = graph.greedy_clique_partition()
    inequalities = set(graph.edges)
    for clique in cliques:
        clique = tuple(sorted(clique))
        for edge in combinations(clique, 2):
            inequalities.remove(edge)
        if len(clique) > 1:
            inequalities.add(clique)

    # Put trivial inequalities for nodes that didn't appear
    # in the constraints, otherwise SCIP will complain
    used_nodes = set()
    for group in inequalities:
        used_nodes.update(group)
    for node in range(10):
        if node not in used_nodes:
            inequalities.add((node,))

    with open(filename, 'w') as lp_file:
        lp_file.write("maximize\nOBJ:" + "".join([f" + 1 x{node+1}" for node in range(len(graph))]) + "\n")
        lp_file.write("\nsubject to\n")
        for count, group in enumerate(inequalities):
            lp_file.write(f"C{count+1}:" + "".join([f" + x{node+1}" for node in sorted(group)]) + " <= 1\n")
        lp_file.write("\nbinary\n" + " ".join([f"x{node+1}" for node in range(len(graph))]) + "\n")

# mis graphs index from 1
def lp2mis(inst_name):

    writeto = inst_name[:-2]+'mis'
    # read problem
    with open(inst_name, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    nnodes = int(lines[-1].split(' ')[-1][1:])
    nconss = int(lines[-4].split(':')[0][1:]) 
    conss = lines[4:-3]
    assert(len(conss) == nconss)

    print(f"\ncurrent graph: {inst_name}, {nnodes}, {nconss}")
    lines = []
    for cons in conss:
        nodes = cons[:-5].split('+')[1:]
        nodes = [node.strip()[1:] for node in nodes]
        for i in range(len(nodes)):
            for j in range(i+1, len(nodes)):
                lines.append(f'e {nodes[i]} {nodes[j]}\n')
    nedges = len(lines)

    with open(writeto, 'w+') as f:
        f.write(f'p edge {nnodes} {nedges}\n')
        for line in lines:
            f.write(line)
    
    return writeto

# snap graphs index from 0
def mis2snap(fpath):

    writeto = fpath[:-3]+'snap'

    with open(fpath, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        nnodes = int(lines[0].split(' ')[2])
        nedges = int(lines[0].split(' ')[3]) 

        lines = lines[1:]
        
        # sort
        def compare(line1, line2):
            from1, to1 = line1[2:].split(" ")
            from1, to1 = int (from1), int(to1)
            from2, to2 = line2[2:].split(" ")
            from2, to2 = int(from2), int(to2)
            if from1 != from2:
                return from1 - from2
            else:
                return to1 - to2

        print(f"\ncurrent graph: {fpath}, {nnodes}, {nedges}")
        with open(writeto, 'w+') as f:
            f.write('# Undirected graph (each unordered pair of nodes is saved once)\n')
            f.write('# Undirected Erdos-Renyi random graph.\n')
            f.write(f'# Nodes: {nnodes} Edges: {nedges}\n')
            f.write(f'# NodeId	NodeId\n')

            numlines = len(lines)
            for idx, line in enumerate(sorted(lines, key=cmp_to_key(compare))):
                node1, node2 = line[2:].split(' ')
                node1, node2 = int(node1), int(node2)
                if idx + 1 == numlines:
                    f.write(f'{node1-1} {node2-1}')
                else:
                    f.write(f'{node1-1} {node2-1}\n')
        return writeto


# metis graphs index from 1
def mis2metis(mis_path):

    writeto = mis_path[:-3]+'metis'

    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [ atoi(c) for c in re.split('(\d+)', text) ]

    number_nodes = 0
    number_edges = 0
    edges_counted = 0
    adjacency = []

    with open(mis_path) as f:
        for line in f:
            args = line.strip().split()

            if args[0] == 'p':
                number_nodes = args[2]
                number_edges = args[3]
                adjacency = [[] for _ in range(0, int(number_nodes) + 1)]
            elif args[0] == 'e':
                source = int(args[1])
                target = int(args[2])
                edge_added = False
                if not target in adjacency[source]:
                    adjacency[source].append(target)
                    edge_added = True
                if not source in adjacency[target]:
                    adjacency[target].append(source)
                    edge_added = True
                if edge_added:
                    edges_counted += 1
            else:
                print ("Could not read line.\n")

    adjacency[0].append(number_nodes)
    # adjacency[0].append(number_edges)
    adjacency[0].append(str(edges_counted))

    with open(writeto, 'w') as f:
        node = 0
        for neighbors in adjacency:
            if node != 0:
                neighbors.sort()
            if not neighbors:
                f.write(' ')
            else:
                tmp = [str(i) for i in neighbors]
                f.write(' '.join(tmp))
            f.write('\n')
            node += 1
    return writeto


def solve_mis(metis_path):
    writeto = f'{metis_path[:-5]}sol'
    os.system(f'./redumis {metis_path} --output={writeto}')

    with open(writeto, 'r') as f:
        sol_content = f.readlines()
        sol_content = [line.strip() for line in sol_content]

    sol_arr = []
    for idx, sol_val in enumerate(sol_content):
        if int(sol_val) == 1:
            sol_arr.append(idx+1)

    with open(writeto, 'w') as f:
        for opt_sol_idx in sol_arr:
            f.write(f'{opt_sol_idx}\n')

    return writeto


# should be snap graph
def save_adj(graph_path, writeto):
    with open(graph_path, 'r') as f:
        graph_content = f.readlines()
        graph_content = [line.strip() for line in graph_content]
        nnodes = int(graph_content[2].split(' ')[2])
        graph_content = graph_content[4:]
    adj_mat = np.zeros((nnodes,nnodes))
    for edge in graph_content:
        ver1, ver2 = edge.split(' ')
        ver1, ver2 = int(ver1), int(ver2)
        adj_mat[ver1, ver2] = 1
        adj_mat[ver2, ver1] = 1

    dic = {
        'adj':adj_mat,
    }

    sio.savemat(writeto, dic)

def gen_mis(data_dir, ninst, scale_lower, scale_upper=None, solve=True):
    
    os.makedirs(data_dir, exist_ok=True)
    affinity = 4

    for i in range(ninst):
        nnodes = scale_lower if scale_upper is None else random.randint(scale_lower, scale_upper+1)
        graph = Graph.barabasi_albert(nnodes, affinity, np.random.RandomState(i))

        lp_path = os.path.join(data_dir, f'{i}.lp')
        generate_indset(graph, lp_path)
        mis_path = lp2mis(lp_path); 
        snap_path = mis2snap(mis_path)
        metis_path = mis2metis(mis_path)
        
        save_adj(snap_path, f'{lp_path[:-2]}adj')

        # solve metis
        if solve:
            sol_path = solve_mis(metis_path)
        os.remove(snap_path); os.remove(metis_path)

if __name__ == '__main__':

    home = expanduser("~")
    data_dir = os.path.join(home, f'storage1/instances/mis/test_2000')
    gen_mis(data_dir, 50, 2000, solve=False)