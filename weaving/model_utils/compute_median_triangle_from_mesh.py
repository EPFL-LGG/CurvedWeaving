elastic_rods_dir = '../../elastic_rods/python/'
import os
import os.path as osp
import sys; sys.path.append(elastic_rods_dir)
import numpy as np, elastic_rods, linkage_vis
import numpy.linalg as la
from elastic_rods import EnergyType, InterleavingType

def compute_median_triangle(input_name, output_name):
    point_list = []
    with open(input_name, 'r') as input_mesh:
        content = input_mesh.readlines()
        for line in content:
            if 'v ' in line:
                point = np.array([float(x) for x in line.split(' ')[1:]])
                point_list.append(point)

        mid_point_list = []
        point_to_mid_point_map = -1 * np.ones((len(point_list), len(point_list)))
        for line in content:
            if 'f ' in line:
                curr_face = np.array([int(x.split('/')[0])-1 for x in line.strip().split(' ')[1:]])
                for i in range(len(curr_face)):
                    v1 = curr_face[i-1]
                    v2 = curr_face[i]
                    v1, v2 = min(v1, v2), max(v1, v2)
                    if point_to_mid_point_map[v1][v2] == -1:
                        mid_point = 0.5 * (point_list[v1] + point_list[v2])
                        point_to_mid_point_map[v1][v2] = len(mid_point_list)
                        mid_point_list.append(mid_point)
        line_list = []
        for line in content:
            if 'f ' in line:
                curr_face = np.array([int(x.split('/')[0])-1 for x in line.strip().split(' ')[1:]])
                median_face = []
                for i in range(len(curr_face)):
                    v1 = curr_face[i-1]
                    v2 = curr_face[i]
                    v1, v2 = min(v1, v2), max(v1, v2)
                    median_face.append(int(point_to_mid_point_map[v1][v2]))
                line_list.extend(([median_face[i-1], median_face[i]] for i in range(len(median_face))))
    with open(output_name, 'w') as output_mesh:
        for vx in mid_point_list:
            output_mesh.write('v {} {} {}\n'.format(vx[0], vx[1], vx[2]))
        for line in line_list:
            output_mesh.write('l {} {}\n'.format(line[0] + 1, line[1] + 1))

    # with open(output_name, 'r') as linkage_file:

    # Project linkage onto target surface
    linkage = elastic_rods.SurfaceAttractedLinkage(target_surface_mesh, False, output_name, 5, False, InterleavingType.weaving)
    vertices = linkage.get_linkage_closest_point()
    vertices = vertices.reshape(int(len(vertices) / 3), 3)
    lines = []
    for si in range(linkage.numSegments()):
        lines.append([linkage.segment(si).startJoint + 1, linkage.segment(si).endJoint + 1])

    with open(output_name, 'w') as output_linkage:
        for vx in vertices:
            output_linkage.write('v {} {} {}\n'.format(vx[0], vx[1], vx[2]))
        for line in lines:
            output_linkage.write('l {} {}\n'.format(line[0], line[1]))


input_name = '../models/heart_wenzel_dual.obj'
output_name = '../models/heart_wenzel.obj'
target_surface_mesh = '../surface_models/heart_wenzel.obj'
compute_median_triangle(input_name, output_name)