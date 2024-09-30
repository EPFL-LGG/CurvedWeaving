elastic_rods_dir = '../../elastic_rods/python/'
import os
import os.path as osp
import sys; sys.path.append(elastic_rods_dir)
import elastic_rods
from elastic_rods import EnergyType, InterleavingType
import numpy as np
import random
def insert_noise_in_linkage(input_name, output_name, target_surface_mesh):
    noisy_points = []
    lines = []
    with open(input_name, 'r') as input_mesh:
        content = input_mesh.readlines()
        for line in content:
            if 'v ' in line:
                point = np.array([float(x) for x in line.split(' ')[1:]])
                vlen = np.array([random.random() for i in range(3)])/20
                noisy_points.append(point + vlen)
            if 'l ' in line:
                lines.append(line)

    with open(output_name, 'w') as linkage_file:
        for vx in noisy_points:
            linkage_file.write('v {} {} {}\n'.format(vx[0], vx[1], vx[2]))
        for line in lines:
            linkage_file.write(line)



    linkage = elastic_rods.SurfaceAttractedLinkage(target_surface_mesh, False, output_name, 5, False, InterleavingType.weaving)
    vertices = linkage.get_linkage_closest_point()
    vertices = vertices.reshape(int(len(vertices) / 3), 3)

    with open(output_name, 'w') as output_linkage:
        for vx in vertices:
            output_linkage.write('v {} {} {}\n'.format(vx[0], vx[1], vx[2]))
        for line in lines:
            output_linkage.write(line)

input_name = '../normalized_objs/models/simple_hemoglobin_5_1.obj'
output_name = '../normalized_objs/models/noisy_simple_hemoglobin_5_1.obj'
target_surface_mesh = '../normalized_objs/surface_models/simple_hemoglobin_5_1.obj'
insert_noise_in_linkage(input_name, output_name, target_surface_mesh)