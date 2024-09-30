import numpy as np 
from numpy import linalg as la

obj_file = "../surface_models/sphere_100mm.obj"
centered_obj_file = '../surface_models/centered_sphere_100mm.obj'

def center_model(obj_file, centered_obj_file):
	point_list = []
	edge_list = []
	face_list = []
	with open(obj_file, 'r') as f, open(centered_obj_file, 'w') as output:
		content = f.readlines()
		count = 0
		edge_count = 0
		for line in content:
			if 'v ' in line:
				point = np.array([float(x) for x in line.split(' ')[1:]])
				point_list.append(point)

			if 'l ' in line:
				edge_list.append(line)

			if 'f ' in line:
				face_list.append(line)

		cm = np.sum(np.array(point_list), axis = 0) / len(point_list)
		# print(cm)

		centered_point_list = []
		point_norm = []
		for point in point_list:
			point = np.array(point) - cm
			centered_point_list.append(point)
			point_norm.append(la.norm(point))
		
		scale = 400.0 / max(point_norm)
		print(scale)
		scale = 262.93249607120157
		scale = 1
		# print(scale, max(point_norm))
		for point in centered_point_list:
			output.write('v {} {} {}\n'.format(scale * point[0], scale * point[1], scale * point[2]))
		for edge in edge_list:
			output.write(edge)

		for face in face_list:
			output.write(face)

obj_file_list = ['/Users/yren/Develop/EPFL_LGG/add_params_elastic_rods/python/weaving_editor/benchmark_result/noOffset/sphere_strip/equal_edge_atI_optimized.obj', '/Users/yren/Develop/EPFL_LGG/add_params_elastic_rods/python/weaving_editor/benchmark_result/noOffset/ellipsoidal_strip/three_axis_ellipsoidal_from_equal_edge_atI_optimized.obj', '/Users/yren/Develop/EPFL_LGG/add_params_elastic_rods/python/weaving_editor/benchmark_result/noOffset/simple_hemoglobin_strip/simple_sphere_to_hemoglobin_optimized.obj', '/Users/yren/Develop/EPFL_LGG/add_params_elastic_rods/python/weaving_editor/benchmark_result/noOffset/regular_torus_strip/regular_torus_optimized.obj', '/Users/yren/Develop/EPFL_LGG/add_params_elastic_rods/python/weaving_editor/benchmark_result/noOffset/gmo_torus_strip/gmo_torus_optimized.obj', '/Users/yren/Develop/EPFL_LGG/add_params_elastic_rods/python/weaving_editor/benchmark_result/noOffset/small_pseudo_sphere_strip/small_pseudo_sphere_optimized.obj']
centered_file_list = ['sphere_opt.obj', 'ellipsoid_opt.obj', 'hemoglobin_opt.obj', 'regular_torus_opt.obj', 'gmo_torus_opt.obj', 'pseudo_sphere_opt.obj']

for obj_file, centered_file in zip(obj_file_list, centered_file_list):
	center_model(obj_file, centered_file)
