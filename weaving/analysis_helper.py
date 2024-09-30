elastic_rods_dir = '../../elastic_rods/python'


import sys; sys.path.append(elastic_rods_dir)
import numpy as np, elastic_rods
import numpy.linalg as la
import matplotlib.cm as cm
from linkage_utils import order_segments_by_ribbons
from linkage_utils import get_turning_angle_and_length_from_ordered_rods

import matplotlib.pyplot as plt
from copy import deepcopy
import scipy
import networkx as nx

def plot_curvatures(original, new = None, esp = 0.001, original_lengths = None, new_lengths = None, remove_zeros = False):
    plt.clf()
    rest_curv = [x if abs(x) >= esp else None for x in original]
    if original_lengths == None:
        plt.plot(rest_curv, label='rest turning angle')
    else:
        accumu_length = [sum(original_lengths[:i+1]) for i in range(len(original_lengths))]
        plt.plot(accumu_length, rest_curv, label='rest turning angle') 
    if new != None:
        curv = [x if abs(x) >= esp else None for x in new]
        if remove_zeros:
            curv = [x if abs(x) > 0 else None for x in new]
        if new_lengths == None:
            plt.plot(curv, label='deformed turning angle')
        else:
            accumu_length = [sum(new_lengths[:i+1]) for i in range(len(new_lengths))]
            plt.plot(accumu_length, curv, label='deformed turning angle') 
    plt.ylabel('turning angle')
    plt.legend()
    plt.show()
    
def compare_turning_angle(linkage, rest_linkage = None, esp = 0.001, remove_zeros = False):
    strips = order_segments_by_strips(linkage)
    angles, lengths, _, _ = get_turning_angle_and_length_from_ordered_rods(strips, linkage)
    if rest_linkage != None:
        rest_angles, rest_lengths, _, _ = get_turning_angle_and_length_from_ordered_rods(strips, rest_linkage, rest = True)
    else:
        rest_angles, rest_lengths, _, _ = get_turning_angle_and_length_from_ordered_rods(strips, linkage, rest = True)

    plot_curvatures(rest_angles[0], angles[0], esp, rest_lengths[0], lengths[0], remove_zeros)

def is_on_sphere(linkage):
    vertices = []
    for i in range(linkage.numSegments()):
        vertices.extend(linkage.segment(i).rod.deformedPoints())
    vertices = np.array(vertices)
    average_point = np.sum(vertices, axis=0) / vertices.shape[0]
    distances = np.array([[la.norm(vx - average_point) for vx in linkage.segment(i).rod.deformedPoints()] for i in range(linkage.numSegments())])
    flat_distances = distances.flatten()
    mean = np.mean(flat_distances)
    std = np.std(flat_distances)
    
    print("The mean distance from the deformed points to the center is {}. The standard deviation is {}.".format(mean, std))
    return distances

def get_distance_to_center_scalar_field(linkage):
    import vis.fields
    distances = is_on_sphere(linkage)
    sf = vis.fields.ScalarField(linkage, distances, colormap=cm.viridis)
    return sf, distances

def get_curvature_scalar_field(linkage):
    import vis.fields
    curvatures = []
    for i in range(linkage.numSegments()):
        curr_kappas = linkage.segment(i).rod.deformedConfiguration().kappa
        curr_kappas = np.array(curr_kappas)[:, 1]
        curr_kappas[0] = curr_kappas[1]
        curr_kappas[-1] = curr_kappas[-2]
        curr_kappas = [kappa ** 2 for kappa in curr_kappas]
        curvatures.extend(curr_kappas)
    sf = vis.fields.ScalarField(linkage, curvatures, colormap=cm.jet)
    return curvatures

def concatenate_rod_properties_from_rod_segments(linkage, rod_strip):
    all_centerline_pos = []
    all_rest_kappas = []
    all_material_frame = []
    all_edge_material = []
    first_segment_flag = True

    # Concatenate the centerline positions, kappa, material frame from all segments in the rod_strip and remove the overlapping end edges at the joints between segments. 
    for elm in rod_strip:
        segment_index = elm[0]
        curr_rod = linkage.segment(segment_index)
        deformed_point = curr_rod.rod.deformedPoints()
        curr_rest_kappas = curr_rod.rod.restKappas()
        curr_material_frame = deepcopy(curr_rod.rod.deformedConfiguration().materialFrame)
        curr_materials = curr_rod.rod.edgeMaterials()
        subdivision_resolution = len(curr_material_frame)
        # If the segment's orientation is fliped, need to reverse the property lists and also negate kappa and the first material axis. 
        if elm[1] == -1:
            deformed_point.reverse()
            curr_rest_kappas.reverse()
            curr_rest_kappas = [kappa * -1 for kappa in curr_rest_kappas]
            curr_materials.reverse()
            curr_material_frame.reverse()
            for frame_index in range(len(curr_material_frame)):
                curr_material_frame[frame_index].d1 = -1 * curr_material_frame[frame_index].d1

        # Keep the overlapping edges between the first and last segments for the case where the rod is a loop.
        if first_segment_flag:
            all_centerline_pos.extend(deformed_point[:subdivision_resolution])
            all_rest_kappas.extend(curr_rest_kappas[:subdivision_resolution])
            all_material_frame.extend(curr_material_frame)
            all_edge_material.extend(curr_materials)
            first_segment_flag = False
        else:
            all_centerline_pos.extend(deformed_point[1:subdivision_resolution])
            all_rest_kappas.extend(curr_rest_kappas[1:subdivision_resolution])
            all_material_frame.extend(curr_material_frame[1:])
            if len(curr_materials) > 1:
                all_edge_material.extend(curr_materials[1:])
            else:
                all_edge_material.extend(curr_materials)

    # Add the final point and its rest curvature.
    all_centerline_pos.append(deformed_point[-1])
    all_rest_kappas.append(curr_rest_kappas[-1])
    # In the case where the rod is a loop:
    # The length of all_centerline_pos should be (subdivision_resolution - 1) * len(rod_strips) + 2.
    # The length of all_rest_kappas should be (subdivision_resolution - 1) * len(rod_strips) + 2.
    # The length of all_material_frame should be (subdivision_resolution - 1) * len(rod_strips) + 1.
    return all_centerline_pos, all_rest_kappas, all_material_frame, all_edge_material

# TODO (Samara): This function only handles the case where the rod is a loop. Should add a flag and allow when the rod is not a loop.
def construct_elastic_rod_loop_from_rod_segments(linkage, rod_strip):
    all_centerline_pos, all_rest_kappas, all_material_frame, all_edge_material = concatenate_rod_properties_from_rod_segments(linkage, rod_strip)
    # Set the centerline pos and initialize thetas to zero.
    new_rod = elastic_rods.ElasticRod(all_centerline_pos)
    # Set the rest kappa to be the same as the rod strip in the linkage.
    new_rod.setRestKappas(all_rest_kappas)
    # Set the edge materials.
    print((len(all_edge_material), len(all_material_frame)))
    if (len(all_edge_material) == len(all_material_frame)):
        print("varying edge material")
        new_rod.setMaterial(all_edge_material)
    else:
        new_rod.setMaterial(all_edge_material[0])
    # Set the reference frame equal to the extracted deformed curve's material frame (and leave thetas at zero).
    for i in range(len(all_material_frame)):
        new_rod.deformedConfiguration().referenceDirectors[i].d1 = all_material_frame[i].d1
        new_rod.deformedConfiguration().referenceDirectors[i].d2 = all_material_frame[i].d2
    new_rod.updateSourceFrame()
    new_rod.setDeformedConfiguration(all_centerline_pos, np.zeros(len(all_material_frame)))

    # If the rod is a loop, then we need to fixed the end edges of the rod by fixing four vertex positions and two thetas.
    last_two_pos = (len(all_centerline_pos) - 2) * 3
    fixedPositionVars = list(range(6)) + list(range(last_two_pos, last_two_pos + 6)) 
    fixedThetaVars = list([x + len(all_centerline_pos) * 3 for x in [0, len(all_material_frame) - 1]])
    fixedVars = fixedPositionVars + fixedThetaVars
    return new_rod, fixedVars

def compute_min_distance_rigid_transformation(input_points, target_points):
    num_points = input_points.shape[0]
    input_cm = np.sum(input_points, axis = 0) / num_points
    target_cm = np.sum(target_points, axis = 0) / num_points
    normalized_input_points = input_points - input_cm
    normalized_target_points = target_points - target_cm
    descriptor_matrix = np.matmul(np.transpose(normalized_target_points), normalized_input_points)
    U, s, Vh = scipy.linalg.svd(descriptor_matrix)
    rotation_matrix = np.matmul(U, Vh)
    if la.det(descriptor_matrix) < 0:
        singular_values = np.identity(3)
        singular_values[2][2] = -1
        rotation_matrix = np.matmul(np.matmul(U, singular_values), Vh)

    translation = np.dot(np.transpose(rotation_matrix), target_cm) - input_cm
    # print(normalized_target_points, np.transpose(np.matmul(rotation_matrix, np.transpose(normalized_input_points))))
    point_wise_difference = normalized_target_points - np.transpose(np.matmul(rotation_matrix, np.transpose(normalized_input_points)))
    # print(point_wise_difference)
    print(sum([la.norm(point) for point in point_wise_difference]))
    transformed_input = np.transpose(np.matmul(rotation_matrix, np.transpose(input_points + translation)))
    return transformed_input, translation, rotation_matrix

def check_max_degree_in_linkage(io):
    import networkx as nx
    G = nx.Graph()
    with open(io.MODEL_PATH, 'r') as linkage:
        content = linkage.readlines()

        edge_list = []
        for line in content:

            if 'l ' in line:
                edge_list.append([int(x) for x in line.strip().split(' ')[1:]])
    G.add_edges_from(edge_list)
    degrees = [key for (value, key) in G.degree]
    return max(degrees)












