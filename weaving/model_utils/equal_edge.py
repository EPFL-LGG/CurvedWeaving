import numpy as np 
from numpy import linalg as la


normalized_obj_file = '../polyhedronisme-atI.obj'
output_file = "partial.obj"

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
	triangle_edge = []
	triangle = None
	for line in content:
		if 'f ' in line:
			count += 1
			line = line.strip()
			face = [int(x.split('/')[0]) for x in line.split(' ')[1:]]
			edge_count += len(face)
			if len(face) == 5:
				print(face)

				# print([point_list[index - 1] for index in face])
				pentagon = np.array([point_list[index - 1] for index in face])
				triangle_edge = face[:2]
				break
	for line in content:
		if 'f ' in line:
			count += 1
			line = line.strip()
			face = [int(x.split('/')[0]) for x in line.split(' ')[1:]]
			edge_count += len(face)
			if len(face) == 3:
				# print([point_list[index - 1] for index in face])
				if triangle_edge[0] in face and triangle_edge[1] in face:
					triangle = [point_list[index - 1] for index in face]

	pentagon_center = np.sum(pentagon, axis = 0) / la.norm(np.sum(pentagon, axis = 0))
	# triangle 1 2 is the common edge
	# triangle 0 and pentagon center is the testing 
	print((pentagon[1], pentagon[0]))
	print((triangle[1], triangle[2]))

	iteration = 100
	i = 0
	curr_pent_one = np.array(pentagon[1])
	curr_pent_two = np.array(pentagon[0])
	curr_end_one = np.array(pentagon_center)
	curr_end_two = np.array(pentagon_center)

	triangle_point = triangle[0] / la.norm(triangle[0])
	while (i < iteration):
		half_one = (curr_end_one + curr_pent_one )/2 / la.norm((curr_end_one + curr_pent_one)/2)
		half_two = (curr_end_two + curr_pent_two )/2 / la.norm((curr_end_two + curr_pent_two)/2)
		if np.dot(half_one, half_two) > np.dot(half_one, triangle_point):
			curr_end_one = half_one
			curr_end_two = half_two
		else:
			curr_pent_one = half_one
			curr_pent_two = half_two
		i += 1
	print(curr_pent_one, curr_pent_two, np.dot(curr_pent_one, curr_pent_two), np.dot(curr_pent_one, triangle_point))
	# alpha is for the vertex point, beta is for the center point
	original_pent_one = np.array(pentagon[1])
	alpha = (curr_pent_one[0] * pentagon_center[1] / pentagon_center[0] - curr_pent_one[1]) / (original_pent_one[0] * pentagon_center[1] / pentagon_center[0] - original_pent_one[1])
	beta = (curr_pent_one[0] - alpha * original_pent_one[0] ) / pentagon_center[0]
	print(curr_pent_one, alpha * original_pent_one + beta * pentagon_center)
	print(alpha, beta)

	point_list = [pentagon[0], pentagon[1], pentagon[2], pentagon[3], pentagon[4], pentagon_center]
	for point in point_list:
		point = 100 * point / la.norm(point)
		point = [point[0], point[1], 2 * point[2]]
		output.write('v {} {} {}\n'.format(point[0], point[1], point[2]))








