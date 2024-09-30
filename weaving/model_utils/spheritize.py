import numpy as np 
from numpy import linalg as la

obj_file = "../polyhedronisme-atI.obj"
normalized_obj_file = './normalized_atl.obj'

point_list = []
edge_list = set()
with open(obj_file, 'r') as f, open(normalized_obj_file, 'w') as output:
	content = f.readlines()
	count = 0
	edge_count = 0
	for line in content:
		if 'v ' in line:
			point = np.array([float(x) for x in line.split(' ')[1:]])
			point_list.append(point)

		if 'f ' in line:
			count += 1
			line = line.strip()
			face = [int(x.split('/')[0]) for x in line.split(' ')[1:]]
			edge_count += len(face)
			for i in range(len(face) - 1):
				edge = face[i:i+2]
				edge.sort()
				edge_list.add(tuple(edge))
			edge = [face[-1], face[0]]
			edge.sort()
			edge_list.add(tuple(edge))
	edge_list = list(edge_list)

	for point in point_list:
		point = 100 * point / la.norm(point)
		point = [point[0], point[1], 2 * point[2]]
		output.write('v {} {} {}\n'.format(point[0], point[1], point[2]))
	for edge in edge_list:
		output.write('l {} {}\n'.format(edge[0], edge[1]))
