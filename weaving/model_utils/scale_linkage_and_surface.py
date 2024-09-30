import numpy as np 
from numpy import linalg as la

linkage_file = "../models/{}.obj"
surface_file = '../surface_models/{}.obj'

output_linkage_file = "../normalized_objs/models/{}.obj"
output_surface_file = '../normalized_objs/surface_models/{}.obj'

def scale_model(target_size, linkage_file, surface_file, output_linkage_file, output_surface_file):
	point_list = []
	edge_list = []
	face_list = []
	with open(linkage_file, 'r') as linkage, open(output_linkage_file, 'w') as output_linkage, open(surface_file, 'r') as surface, open(output_surface_file, 'w') as output_surface:
		content = linkage.readlines()
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

		centered_point_list = []
		point_norm = []
		for point in point_list:
			point = np.array(point) - cm
			centered_point_list.append(point)
			point_norm.append(la.norm(point))
		
		scale = target_size / max(point_norm)
		print(scale, cm)

		for line in content:
			if 'v ' in line:
				point = np.array([float(x) for x in line.split(' ')[1:]])
				point = scale * (point - cm)
				output_linkage.write('v {} {} {}\n'.format(point[0], point[1], point[2]))
			else:
				output_linkage.write(line)

		surface_content = surface.readlines()
		for line in surface_content:
			if 'v ' in line:
				point = np.array([float(x) for x in line.split(' ')[1:]])
				point = scale * (point - cm)
				output_surface.write('v {} {} {}\n'.format(point[0], point[1], point[2]))
			else:
				output_surface.write(line)

# linkage_name = 'simple_sphere_to_hemoglobin/5'
# surface_name = 'sphere_to_hemoglobin/5'
# linkage_name = 'small_pseudo_sphere'
# surface_name = 'pseudo_sphere/small_pseudo_sphere'

# for target_size in [0.1, 1, 10, 50, 100, 200, 400]:
# 	scale_model(target_size, linkage_file.format(linkage_name), surface_file.format(surface_name), output_linkage_file.format(linkage_name+'_{}'.format(str(target_size))), output_surface_file.format(surface_name+'_{}'.format(str(target_size))))

# linkage_name = 'freeform_5'
# surface_name = 'freeform_5'

# linkage_name = 'regular_torus'
# surface_name = 'regular_torus'

# linkage_name = 'lounge_pod'
# surface_name = 'lounge_pod'

# linkage_name = 'bird_open_beak'
# surface_name = 'bird'

# linkage_name = 'fancy_chair_dense'
# surface_name = 'fancy_chair'

# linkage_name = 'lounge_pod'
# surface_name = 'lounge_pod'

# linkage_name = 'heart_coarse'
# surface_name = 'heart_coarse'

# linkage_name = 'surprise_heart'
# surface_name = 'surprise_heart'

# linkage_name = 'flying_bird'
# surface_name = 'flying_bird'

# linkage_name = 'hanging_lamp'
# surface_name = 'hanging_lamp'

# linkage_name = 'nest_chair'
# surface_name = 'nest_chair'

# linkage_name = 'fuzzy_nest_chair'
# surface_name = 'fuzzy_nest_chair'

linkage_name = 'regular_torus'
surface_name = 'regular_torus'

# linkage_name_list = ['sphere', 'ellipsoidal', 'regular_torus', 'gmo_torus', 'small_pseudo_sphere', 'simple_hemoglobin_5', 'bird_close_beak', 'heart_coarse', 'flying_bird', 'nest_chair', 'lounge_pod', 'hanging_lamp', 'kleinbottle']
# linkage_name_list = ['simple_hemoglobin_5', 'simple_hemoglobin_1', 'simple_hemoglobin_2', 'simple_hemoglobin_3', 'simple_hemoglobin_4']
linkage_name_list = ['clam']
surface_name_list = linkage_name_list
target_size = 100
for linkage_name, surface_name in zip(linkage_name_list, surface_name_list):
	scale_model(target_size, linkage_file.format(linkage_name), surface_file.format(surface_name), output_linkage_file.format(linkage_name+'_{}'.format(str(target_size))), output_surface_file.format(surface_name+'_{}'.format(str(target_size))))
