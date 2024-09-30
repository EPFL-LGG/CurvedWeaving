alpha = 0.883899443127722 
beta = 0.12240357370826947

import numpy as np 
from numpy import linalg as la


normalized_obj_file = '../models/polyhedronisme-atI.obj'
output_file = "../models/three_axis_ellipsoidal_equal_edge_atI.obj"

point_list = []
edge_list = set()
with open(normalized_obj_file, 'r') as f, open(output_file, 'w') as output:
	content = f.readlines()
	count = 0
	edge_count = 0
	for line in content:
		if 'v ' in line:
			point = np.array([float(x) for x in line.split(' ')[1:]])
			point_list.append(point / la.norm(point))

	pentagon = None
	for line in content:
		if 'f ' in line:
			count += 1
			line = line.strip()
			face = [int(x.split('/')[0]) for x in line.split(' ')[1:]]
			edge_count += len(face)
			if len(face) == 5:
				pentagon = np.array([point_list[index - 1] for index in face])
				pentagon_center = np.sum(pentagon, axis = 0) / la.norm(np.sum(pentagon, axis = 0))
				for index in face:
					new_point = alpha * point_list[index - 1] + beta * pentagon_center
					new_point /= la.norm(new_point)
					point_list[index - 1] = new_point
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
		point = [point[0], 3*point[1], 1.5*point[2]]
		output.write('v {} {} {}\n'.format(point[0], point[1], point[2]))
	for edge in edge_list:
		output.write('l {} {}\n'.format(edge[0], edge[1]))

	dis_list = []
	for edge in edge_list:
		point_one = point_list[edge[0] - 1]
		point_two = point_list[edge[1] - 1]
		dis = la.norm(point_one - point_two)
		dis_list.append(dis)
	print(np.mean(dis_list), np.std(dis_list))


