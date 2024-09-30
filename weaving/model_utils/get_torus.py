import numpy as np 
from numpy import linalg as la
import networkx as nx

normalized_obj_file = '../hex_torus_10_6.obj'
output_file = "../tri_hex_torus_linkage_10_6.obj"

point_list = []
edge_list = set()
with open(normalized_obj_file, 'r') as f, open(output_file, 'w') as output:
    content = f.readlines()
    content = [line.rstrip() for line in content]
    count = 0
    average_norm = 0
    for line in content:
        if 'v ' in line:
            point = np.array([float(x) for x in line.split()[1:]])
            point_list.append(point)
            average_norm += la.norm(point)
    average_norm /= len(point_list)
    point_list = [pt / average_norm * 100 for pt in point_list]

    for line in content:
        if 'f ' in line:
            count += 1
            line = line.strip()
            face = [int(x.split('/')[0]) for x in line.split()[1:]]
            for i in range(len(face) - 1):
                edge = face[i:i+2]
                edge.sort()
                edge_list.add(tuple(edge))
            edge = [face[-1], face[0]]
            edge.sort()
            edge_list.add(tuple(edge))

    edge_list = list(edge_list)
    G=nx.Graph()
    G.add_edges_from(edge_list)
    for node in G.nodes():
        node_edge_list = list(G.edges(node))
        edge_len_list = []
        for edge in node_edge_list:
            pt_one = point_list[edge[0]-1]
            pt_two = point_list[edge[1]-1]
            edge_len = la.norm(pt_one - pt_two)
            edge_len_list.append(edge_len)
        min_edge_len = min(edge_len_list)
        for i in range(len(node_edge_list)):
            if edge_len_list[i] > min_edge_len * 1.5:
                G.remove_edge(*node_edge_list[i])
        print(G.edges(node))
    for point in point_list:
        output.write('v {} {} {}\n'.format(point[0], point[1], point[2]))
    for edge in G.edges():
        output.write('l {} {}\n'.format(edge[0], edge[1]))

