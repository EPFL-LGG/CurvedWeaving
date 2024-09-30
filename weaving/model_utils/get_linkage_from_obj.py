import numpy as np 
from numpy import linalg as la


normalized_obj_file = '/Users/yingyingren/Develop/EPFL_LGG/AuxeticsCode2019/TestModels/squidward_positive_praux_3D.obj'
output_file = "../models/squidward_linkage.obj"

point_list = []
edge_list = set()
with open(normalized_obj_file, 'r') as f, open(output_file, 'w') as output:
	content = f.readlines()
	count = 0
	average_norm = 0
	for line in content:
		if 'v ' in line:
			point = np.array([float(x) for x in line.split(' ')[1:]])
			point_list.append(point)
	centroid = np.sum(np.array(point_list), axis = 0) / len(point_list)
	norms = np.array([la.norm(point - centroid) for point in point_list])
	average_norm = np.sum(norms) / len(point_list)
	point_list = [pt / average_norm * 100 for pt in point_list]

	for line in content:
		if 'f ' in line:
			count += 1
			line = line.strip()
			face = [int(x.split('/')[0]) for x in line.split(' ')[1:]]
			for i in range(len(face) - 1):
				edge = face[i:i+2]
				edge.sort()
				edge_list.add(tuple(edge))
			edge = [face[-1], face[0]]
			edge.sort()
			edge_list.add(tuple(edge))

	edge_list = list(edge_list)

	for point in point_list:
		output.write('v {} {} {}\n'.format(point[0], point[1], point[2]))
	for edge in edge_list:
		output.write('l {} {}\n'.format(edge[0], edge[1]))

