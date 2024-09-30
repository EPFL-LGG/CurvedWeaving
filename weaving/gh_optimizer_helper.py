elastic_rods_dir = '../elastic_rods/python/'
weaving_dir = './'

import importlib
import os
import os.path as osp
import sys

sys.path.append(elastic_rods_dir)
sys.path.append(weaving_dir)
import json

from math import acos
from math import sqrt
from math import pi

import elastic_rods
from elastic_rods import InterleavingType
import numpy as np
from scipy.spatial import KDTree
import numpy.linalg as la
from linkage_utils import order_segments_by_ribbons

import cross_section_scaling
importlib.reload(cross_section_scaling)
        
class WeavingIO:
    def __init__(self, filename):
        ################################################
        ########### OPEN JSON FILE
        ################################################
        with open(filename) as json_file:
            data = json.load(json_file)

        ################################################
        ########### PARSE OPTIMIZATION SETTINGS
        ################################################
        optimization_options = data['OptimizationOptions']
        auto_scale_width = bool(optimization_options['AutomaticVariedCrossSection'])
        min_scaling_factor = float(optimization_options['MinWidthScalingFactor'])
        max_scaling_factor = float(optimization_options['MaxWidthScalingFactor'])
        self.num_optimization_stages = int(optimization_options['NumOptimizationStages'])
        self.number_of_updates = int(optimization_options['NumberOfUpdates'])
        self.update_attraction_weight = float(optimization_options['UpdateAttractionWeight'])
        
        ################################################
        ########### INTERLEAVING
        ################################################
        itype = data['Interleaving']
        self.interleaving = InterleavingType.triaxialWeave
        if (itype == 1):
            self.interleaving = InterleavingType.weaving
            
        ################################################
        ########### PARSE GRAPH
        ################################################
        graph = data['Graph']
        nodes_data = graph['Nodes']
        normals_data = graph['Normals']
        edges_data = graph['EdgeIndices']
        num_nodes = graph['NumNodes']
        num_edges = graph['NumEdges']
        
        ################################################
        ########### NODES
        ################################################
        nodes = np.ndarray(shape=(num_nodes, 3), dtype=np.float64)
        nodes_normals = np.ndarray(shape=(num_nodes, 3), dtype=np.float64)
        for i in range(num_nodes):
            pos = nodes_data[i]
            nodes[i] = [np.float64(pos['X']), np.float64(pos['Y']), np.float64(pos['Z'])]
            norm = normals_data[i]
            nodes_normals[i] = [np.float64(norm['X']), np.float64(norm['Y']), np.float64(norm['Z'])]
            
        ################################################
        ########### EDGES
        ################################################
        edges = np.ndarray(shape=(num_edges, 2), dtype=np.int32)
        self.subdivision = np.int32(np.average([subd for subd in graph['EdgeSubdivisions']]))
        for i in range(num_edges):
            e = edges_data[i]
            edges[i] = [np.int32(e[0]), np.int32(e[1])]
            
        ################################################
        ########### INIT. LINKAGE
        ################################################
        try:
            self.linkage = elastic_rods.RodLinkage(nodes, edges, self.subdivision, initConsistentAngle=False, rod_interleaving_type=self.interleaving)
            self.isValid = True
        except RuntimeError as txt:
            print('[ERROR] Invalid inputs for initialization!')
            print(txt)
            self.isValid = False
            return 
          
        joints = KDTree(np.array([ jt.position for jt in self.linkage.joints()]))
        
        ################################################
        ########### MATERIALS
        ################################################
        materials = data['Materials']
        num_materials = len(materials)
        m = materials[0]
        self.ref_width = m['Parameters'][0]
        self.ref_height = m['Parameters'][1]
        
        if auto_scale_width:
            m = materials[0]
            section, self.isValid  = self.parse_cross_section(m['CrossSectionType'])
            mat0 = elastic_rods.CrossSection.construct(section, m['E'], m['PoisonsRatio'], [m['Parameters'][0] * min_scaling_factor, m['Parameters'][1]]) 
            mat1 = elastic_rods.CrossSection.construct(section, m['E'], m['PoisonsRatio'], [m['Parameters'][0] * max_scaling_factor, m['Parameters'][1]]) 
            cross_section_scaling.apply_density_based_cross_sections(self.linkage, mat0, mat1)
        else:
            if num_materials == num_nodes:
                mats = [-1] * num_nodes
                for m in materials:
                    section, self.isValid = self.parse_cross_section(m['CrossSectionType'])
                    # Reference point
                    pos = m['ReferencePosition']
                    # Query the KDTree for the closest point
                    distance, index = joints.query([np.float64(pos['X']), np.float64(pos['Y']), np.float64(pos['Z'])])
                    # Assign material to that joint
                    mats[index] = elastic_rods.RodMaterial(section, m['E'], m['PoisonsRatio'], [m['Parameters'][0], m['Parameters'][1]])
                self.linkage.setJointMaterials(mats)
            else:
                m = materials[0]
                section, self.isValid = self.parse_cross_section(m['CrossSectionType'])
                mat = elastic_rods.RodMaterial(section, m['E'], m['PoisonsRatio'], [m['Parameters'][0], m['Parameters'][1]])
                self.linkage.setMaterial(mat)
        
        # Check if the cross-sections are valid
        if self.isValid==False:
            print('[ERROR] The cross-section used for the woven model is invalid. Please note that only rectangle and ellipse cross-sections are allowed.')
            return 
        
        ################################################
        ########### SUPPORTS
        ################################################
        supports = data['Supports']
        self.fixed_vars = []
        for sp in supports:
            # Reference point
            pos = sp['ReferencePosition']
            locked_dofs = sp['LockedDoFs']
            # Query the KDTree for the closest point
            distance, index = joints.query([np.float64(pos['X']), np.float64(pos['Y']), np.float64(pos['Z'])])
            index_offset = self.linkage.dofOffsetForJoint(index)
            
            dofs = [index_offset + i for i in range(len(locked_dofs)) if bool(locked_dofs[i])]
            self.fixed_vars.extend(dofs)
        
        ################################################
        ########### TARGET SURFACE
        ################################################
        try:
            target_surface = data['TargetSurface']
                    
            target_vertices = target_surface['Vertices']
            num_target_vertices = len(target_vertices)
            self.target_vertices = np.ndarray(shape=(num_target_vertices, 3), dtype=np.float64)
            for i in range(num_target_vertices):
                v = target_vertices[i]
                self.target_vertices[i] = [float(v[0]), float(v[1]), float(v[2])]

            target_faces = target_surface['Trias']
            num_target_faces = len(target_faces)
            self.target_faces = np.ndarray(shape=(num_target_faces, 3), dtype=np.int32)
            for i in range(num_target_faces):
                f = target_faces[i]
                self.target_faces[i] = [int(f[0]), int(f[1]), int(f[2])]

            self.surface_attracted_linkage = elastic_rods.SurfaceAttractedLinkage(target_vertices=self.target_vertices, target_faces=self.target_faces, useCenterline=True, rod=self.linkage)
            self.surface_attracted_linkage.set_holdClosestPointsFixed(True)
            self.surface_attracted_linkage.set_attraction_tgt_joint_weight(0.01)
            self.surface_attracted_linkage.attraction_weight = 100
            
            print('Woven Model Constructed\n')
        except Exception as e:
            self.surface_attracted_linkage = None 
            print(f"An error occurred during the construction of the target surface: {e}")


    def parse_cross_section(self, type : int):
        if (type == 0):
            section = 'rectangle'
            is_valid = True
        elif (type == 1):
            section = 'ellipse'
            is_valid = True
        else:
            section = None
            is_valid = False
        return section, is_valid

def generate_grasshopper_data_gh(model_name: str):
    filename = osp.join(weaving_dir + 'grasshopper/inputs/{}.json'.format(model_name))
    with open(filename) as json_file:
        data = json.load(json_file)
        num_optimization_stages = int(data['OptimizationOptions']['NumOptimizationStages'])

    data_root = 'grasshopper/optimization_diagram_results'
    output_root = 'grasshopper/outputs'

    import pickle, gzip
    stage_1_pickle_name = os.path.join(data_root, '{}/{}_stage_1.pkl.gz'.format(model_name, model_name))
    stage_2_pickle_name = os.path.join(data_root, '{}/{}_stage_2.pkl.gz'.format(model_name, model_name))
    stage_3_pickle_name = os.path.join(data_root, '{}/{}_stage_3.pkl.gz'.format(model_name, model_name))

    if num_optimization_stages == 1:
        linkage = pickle.load(gzip.open(stage_1_pickle_name, 'r'))
    elif num_optimization_stages == 2:
        linkage = pickle.load(gzip.open(stage_2_pickle_name, 'r'))
    else:
        linkage = pickle.load(gzip.open(stage_3_pickle_name, 'r'))

    write_optimization_info_json(linkage, '{}/{}_optimization.json'.format(output_root, model_name))


def write_optimization_info_json(linkage, filename):
    top_x_list, top_y_list, bottom_x_list, bottom_y_list, joint_x_list, joint_y_list, all_joint_index, rods_vertices, rods_faces, ribbon_joints, center_line_x_list, center_line_y_list, ribbons_to_segments = write_per_ribbon(linkage)

    def get_crossing_json(pos, ribbon_pair):
        return {'position': list(pos), 'ribbons': ribbon_pair}

    def get_ribbon_crossing_list(index):
        selected_list = []
        selected_ribbon = []
        for ribbon_index, index_list in enumerate(flat_joint_indexes):
            if index in set(index_list):
                selected_ribbon.append(ribbon_index)
                selected_list.append(index_list)
        return selected_ribbon
    
    def get_position_list(idx, data_x, data_y, data_xx=None, data_yy=None):
        pos_list = [ [data_x[idx][i],data_y[idx][i], 0] for i in range(len(data_x[idx]))]
        if data_xx != None: pos_list.extend([data_xx[idx][i],data_yy[idx][i],0] for i in range(len(data_xx[idx])-1, -1, -1))
        return pos_list

    def get_joints_list(idx, data):
        j_list = [ [float(pos[0]), float(pos[1]), float(pos[2])] for pos in data[idx] ]
        return j_list
    
    def get_mesh_vertices_list(idx, data):
        v_list = [ [ [float(pos[0]), float(pos[1]), float(pos[2])] for pos in seg]  for seg in data[idx] ]
        return v_list
    
    def get_mesh_quad_faces_list(idx, data):
        f_list = [ [ [int(face[0]), int(face[1]), int(face[2]), int(face[3])] for face in seg ] for seg in data[idx] ]
        return f_list

    flat_ribbons_outlines = [ get_position_list(idx, top_x_list, top_y_list, bottom_x_list, bottom_y_list) for idx in range(len(top_x_list)) ]
    flat_joint_positions = [ get_position_list(idx, joint_x_list, joint_y_list) for idx in range(len(top_x_list)) ]
    flat_center_line = [ get_position_list(idx, center_line_x_list, center_line_y_list) for idx in range(len(top_x_list)) ]
    flat_joint_indexes = [j_list + [j_list[0]] for j_list in all_joint_index]

    deformed_linkage_mesh_vertices = [ get_mesh_vertices_list(idx, rods_vertices)  for idx in range(len(rods_vertices))]
    deformed_linkage_mesh_faces = [ get_mesh_quad_faces_list(idx, rods_faces)  for idx in range(len(rods_faces))]
    deformed_linkage_joints = [ get_joints_list(idx, ribbon_joints) for idx in range(len(ribbon_joints))]

    pairs_of_ribbons_per_crossing = [ get_ribbon_crossing_list(i) for i in range(linkage.numJoints())]
    crossing_positions = linkage.jointPositions().reshape((linkage.numJoints(), 3))

    crossing_info_list = [get_crossing_json(crossing_positions[c_index], pairs_of_ribbons_per_crossing[c_index])
                          for c_index in range(len(pairs_of_ribbons_per_crossing))]

    # 3D centerlines
    ribbons_centerlines = []
    for ribbon in ribbons_to_segments:
        ribbons_seg = []
        for seg_index in ribbon:
            seg = linkage.segment(seg_index)
            pts = []
            pts_data = seg.rod.restPoints()
            num_pts = len(pts_data)
            for i in range(num_pts):
                if(i==0 and seg.hasStartJoint()):
                    p0 = pts_data[0]
                    p1 = pts_data[1]
                    pts.append([float((p0[0]+p1[0])*0.5), float((p0[1]+p1[1])*0.5), float((p0[2]+p1[2])*0.5)])
                elif(i==num_pts-1 and seg.hasEndJoint()):
                    p0 = pts_data[-1]
                    p1 = pts_data[-2]
                    pts.append([float((p0[0]+p1[0])*0.5), float((p0[1]+p1[1])*0.5), float((p0[2]+p1[2])*0.5)])
                else:
                    p = pts_data[i]
                    pts.append([float(p[0]), float(p[1]), float(p[2])])
            ribbons_seg.append(pts)
        ribbons_centerlines.append(ribbons_seg)
            
    model_info = {'crossings': crossing_info_list, 'flat_ribbon_outlines' : flat_ribbons_outlines, 'flat_joint_positions' : flat_joint_positions, 'flat_joint_indexes' : flat_joint_indexes, 'flat_center_line' : flat_center_line, 
                  'deformed_rod_vertices' : deformed_linkage_mesh_vertices,'deformed_rod_quad_faces' : deformed_linkage_mesh_faces, 'deformed_joints' : deformed_linkage_joints, 'deformed_center_line' : ribbons_centerlines, 'number_joints' : int(linkage.numJoints()) }
    
    with open(filename, 'w') as f:
        json.dump(model_info, f, indent=4)
    print("Output generated")


def write_per_ribbon(linkage, rod_resolution=10, flip_angles = False, bending_axis = 0):
    ribbons = order_segments_by_ribbons(linkage)
    all_ribbon_angle, all_ribbon_edge_len, all_ribbon_num_seg, all_ribbon_widths, all_joint_index, all_extra_first_edge, all_segment_index = get_turning_angle_and_length_from_ordered_rods_2(ribbons, linkage, rest = True, bending_axis = bending_axis)
    if flip_angles:
        all_ribbon_angle = [[w * -1 for w in row] for row in all_ribbon_angle]

    # Parameters for fabrication
    all_ribbon_widths = [[w / 2 for w in row] for row in all_ribbon_widths]

    top_x_list, top_y_list, bottom_x_list, bottom_y_list, joint_x_list, joint_y_list, ribbon_index_list, all_joint_angle_list, all_joint_direction_list, upper_joint_x_list, upper_joint_y_list, lower_joint_x_list, lower_joint_y_list = [], [], [], [], [], [], [], [], [], [], [], [], []
    rod_vertices, rod_faces, ribbon_joints = [],[],[]
    center_line_x_list, center_line_y_list  = [], []
    select_ribbon_index = range(len(all_ribbon_angle))

    ribbons_to_segments = []
    for ribbon_index in select_ribbon_index:
        top_x, top_y, bottom_x, bottom_y, joint_x, joint_y, joints_angle, joints_direction, upper_joints_xs, upper_joints_ys, lower_joints_xs, lower_joints_ys, center_line_x, center_line_y = get_laser_cutting_pattern_data(ribbons, all_ribbon_angle, all_ribbon_edge_len, all_ribbon_num_seg, all_ribbon_widths, rod_resolution, ribbon_index = ribbon_index)

        top_x_list.append(top_x) 
        top_y_list .append(top_y) 
        bottom_x_list.append(bottom_x) 
        bottom_y_list.append(bottom_y) 
        joint_x_list.append(joint_x) 
        joint_y_list.append(joint_y) 
        ribbon_index_list.append(ribbon_index) 
        all_joint_angle_list.append(joints_angle)
        all_joint_direction_list.append(joints_direction)
        upper_joint_x_list.append(upper_joints_xs) 
        upper_joint_y_list.append(upper_joints_ys) 
        lower_joint_x_list.append(lower_joints_xs) 
        lower_joint_y_list.append(lower_joints_ys) 
        center_line_x_list.append(center_line_x)
        center_line_y_list.append(center_line_y)

        rod_vertices.append([linkage.segment(seg_index).rod.rawVisualizationGeometry(True,True)[0] for seg_index in all_segment_index[ribbon_index]])
        rod_faces.append([linkage.segment(seg_index).rod.rawVisualizationGeometry(True,True)[1] for seg_index in all_segment_index[ribbon_index]])
        ribbon_joints.append([linkage.joint(jt_index).position for jt_index in all_joint_index[ribbon_index]])
        ribbons_to_segments.append([seg_index for seg_index in all_segment_index[ribbon_index]])

    return top_x_list, top_y_list, bottom_x_list, bottom_y_list, joint_x_list, joint_y_list, all_joint_index, rod_vertices, rod_faces, ribbon_joints, center_line_x_list, center_line_y_list, ribbons_to_segments 

def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return np.array([qx, qy])

def length(v):
    return sqrt(v[0]**2+v[1]**2)

def dot_product(v,w):
    return v[0]*w[0]+v[1]*w[1]

def determinant(v,w):
    return v[0]*w[1]-v[1]*w[0]

def inner_angle(v,w):
    cosx=dot_product(v,w)/(length(v)*length(w))
    rad=acos(cosx) # in radians
    return rad*180/pi # returns degrees

def angle_clockwise(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner / 180 * np.pi
    else: # if the det > 0 then A is immediately clockwise of B
        return (360-inner) / 180 * np.pi

def angle_counterclockwise(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det>0: #this is a property of the det. If the det > 0 then B is counterclockwise of A
        return inner / 180 * np.pi
    else: # if the det > 0 then A is immediately clockwise of B
        return (360-inner) / 180 * np.pi

def get_curve_from_angle(angles, edge_lens, widths, segment_res, with_extra_tip = False):
    if with_extra_tip:
        angles = angles[-int(segment_res / 2) :] + angles + angles[:int(segment_res / 2)]
        edge_lens = edge_lens[-int(segment_res / 2) :] + edge_lens + edge_lens[:int(segment_res / 2)]
        widths = widths[-int(segment_res / 2) :] + widths + widths[:int(segment_res / 2)]
    curr_end_point = np.array([0, 0])
    curr_angle = 0
    curve_point_list = []
    for i in range(len(angles)):
        curr_angle += angles[i]
        curr_angle %= 2 * np.pi
        edge = np.array([edge_lens[i], 0]) + curr_end_point
        curr_end_point = rotate(curr_end_point, edge, curr_angle)
        curve_point_list.append(curr_end_point)
    curve_point_list = np.array(curve_point_list)
    # min_index = np.array(angles).argmin()
    # second_min_index = np.argpartition(np.array(angles), 1)[1]
    rotated_vector = curve_point_list[-1] - curve_point_list[0]
    # Module by pi so that the curve's x coordinates goes from small to large
    rotated_angle = angle_counterclockwise(rotated_vector, [1, 0])
    curve_point_list = np.array([rotate([0, 0], point, rotated_angle) for point in curve_point_list])
    xs, ys = curve_point_list[:, 0], curve_point_list[:, 1]

    return xs, ys, widths

def get_laser_cutting_pattern_data(ribbons, all_ribbon_angle, all_ribbon_edge_len, all_ribbon_num_seg, all_ribbon_widths, resolution, ribbon_index = 0):
    rod_angle, rod_length, num_seg_per_rod, rod_width = all_ribbon_angle[ribbon_index], all_ribbon_edge_len[ribbon_index], all_ribbon_num_seg[ribbon_index], all_ribbon_widths[ribbon_index]
    xs, ys, widths = get_curve_from_angle(rod_angle, rod_length, rod_width, resolution, True)
    # print("ribbon centerline: ", xs, ys)
    # print("scale: ", scale)
    print("num seg per rod", num_seg_per_rod)
    # print("num ribbons", len(ribbons))
    # print("rod length size: ", len(rod_length))
    # print("rod length: ", [sum(rod_length[i * (resolution - 1) : (i + 1) * (resolution - 1)]) for i in range(num_seg_per_rod-1)])
    # print(min([sum(rod_length[i * (resolution - 1) : (i + 1) * (resolution - 1)]) for i in range(num_seg_per_rod-1)]) * scale)
    
    # Assume the rod are uniformly sampled.
    point_per_segment = int(len(all_ribbon_edge_len[ribbon_index]) / num_seg_per_rod)

    def scale_to_frame(xs, ys):
        scaled_xs = np.array([(x - min(xs)) for x in xs])
        scaled_ys = np.array([y for y in ys])
        return scaled_xs, scaled_ys
    
    def get_point(index):
        return np.array([scaled_xs[index], scaled_ys[index]])
    
    def get_extension(index):
        return extend_direction_list[index] * widths[index]

    def rotate_90_ccw(vec):
        x, y = vec[0], vec[1]
        angle = np.pi / 2
        n_x = x * np.cos(angle) + y * np.sin(angle)
        n_y = -x * np.sin(angle) + y * np.cos(angle)
        return np.array([n_x, n_y])
    
    scaled_xs, scaled_ys = scale_to_frame(xs, ys)
    extend_direction_list = []
    for i in range(len(scaled_ys))[1:-1]:
        prev_point, next_point, curr_point = get_point(i-1), get_point(i+1), get_point(i)
        to_prev, to_next = prev_point - curr_point, next_point - curr_point
        to_prev /= la.norm(to_prev)
        to_next /= la.norm(to_next)
        extend_direction = to_prev + to_next
        if la.norm(extend_direction) <= 1e-7:
            extend_direction = rotate_90_ccw(to_next)
        extend_direction /= la.norm(extend_direction)
        correct_sign = np.sign(np.cross(to_next, extend_direction))
        if correct_sign == 0:
            print("Warning: the centerline is exactly straight.")
        extend_direction *= correct_sign
        extend_direction_list.append(extend_direction)
    extend_direction_list = extend_direction_list[:1] + extend_direction_list + extend_direction_list[-1:]
    upper_point_list, lower_point_list, center_point_list = [], [], []
    for i in range(len(extend_direction_list)):
        curr_point = get_point(i)      
        upper_point = curr_point + get_extension(i)
        lower_point = curr_point - get_extension(i)
        upper_point_list.append(upper_point)
        lower_point_list.append(lower_point)
        center_point_list.append(curr_point)
    upper_point_list, lower_point_list, center_point_list = np.array(upper_point_list), np.array(lower_point_list), np.array(center_point_list)

#     The joint position since the joint is in the middle of the segments
#     Assume the rod segment are concatenated together to form the longer rod, the joint positions are at the start of each segment. 
    tip_size = int(resolution / 2)
    joints = np.array([(get_point(i * point_per_segment + tip_size - 1) + get_point(i * point_per_segment - 2 + tip_size)) / 2 for i in range(num_seg_per_rod + 1)])
    upper_joints = np.array([(get_point(i * point_per_segment + tip_size - 1) + get_point(i * point_per_segment - 2 + tip_size) + get_extension(i * point_per_segment + tip_size - 1) + get_extension(i * point_per_segment - 2 + tip_size)) / 2  for i in range(num_seg_per_rod + 1)])
    lower_joints = np.array([(get_point(i * point_per_segment + tip_size - 1) + get_point(i * point_per_segment - 2 + tip_size) - get_extension(i * point_per_segment + tip_size - 1) - get_extension(i * point_per_segment - 2 + tip_size)) / 2  for i in range(num_seg_per_rod + 1)])


    joints_angle = [angle_clockwise(np.array([1, 0]), get_point(i * point_per_segment + tip_size - 1) - get_point(i * point_per_segment - 2 + tip_size)) for i in range(num_seg_per_rod + 1)] 
    joints_direction = [get_point(i * point_per_segment + tip_size - 1 + int(tip_size / 4)) - get_point(i * point_per_segment - 1 - int(tip_size / 4) + tip_size) for i in range(num_seg_per_rod + 1)] 
    joints_direction = [d / la.norm(d) for d in joints_direction]

    top_x = upper_point_list[:, 0]
    top_y = upper_point_list[:, 1]
    bottom_x = lower_point_list[:, 0]
    bottom_y = lower_point_list[:, 1]
    center_x = center_point_list[:, 0]
    center_y = center_point_list[:, 1]
    joints_xs, joints_ys = joints[:, 0], joints[:, 1]
    upper_joints_xs, upper_joints_ys = upper_joints[:, 0], upper_joints[:, 1]
    lower_joints_xs, lower_joints_ys = lower_joints[:, 0], lower_joints[:, 1]

#     np.savez('ground_truth_sphere_laser_cutting_pattern.npz', top_x = np.array(top_x), top_y = np.array(top_y), bottom_x = np.array(bottom_x), bottom_y = np.array(bottom_y), joints_xs = np.array(joints_xs), joints_ys = np.array(joints_ys))
    return top_x, top_y, bottom_x, bottom_y, joints_xs, joints_ys, joints_angle, joints_direction, upper_joints_xs, upper_joints_ys, lower_joints_xs, lower_joints_ys, center_x, center_y

def get_turning_angle_and_length_from_ordered_rods_2(ribbons, linkage, rest = False, bending_axis = 0):
    ''' This function takes the output of the function "order_segments_by_ribbons" and extract curvatures and length information. The information is organized in the same format as that function.
    '''
    all_ribbon_angles = []
    all_ribbon_edge_len = []
    all_extra_first_edge = []
    all_ribbon_num_seg = []
    all_ribbon_joints = []
    all_ribbon_widths = []
    all_segment_index = []
    for curr_ribbon in ribbons:
        all_ribbon_num_seg.append(len(curr_ribbon))
        geo_curvatures = []
        edge_len = []
        edge_width = []
        joint_index = []
        segment_index = []
        # Check whether the rod is a loop
        last_segment = curr_ribbon[-1]
        extra_first_edge = None
        if (last_segment[1] == 1 and linkage.segment(last_segment[0]).endJoint == None) or (last_segment[1] == -1 and linkage.segment(last_segment[0]).startJoint == None):

            if rest:
                extra_first_edge = linkage.segment(curr_ribbon[0][0]).rod.restLengths()[0]
            else:
                extra_first_edge = linkage.segment(curr_ribbon[0][0]).rod.deformedConfiguration().len[0]
        all_extra_first_edge.append(extra_first_edge)

        for i in range(len(curr_ribbon)):
            seg_index = curr_ribbon[i][0]
            correct_orientation = curr_ribbon[i][1] == 1
            curr_curvature = None
            r = linkage.segment(seg_index).rod
            segment_index.append(seg_index)

            if rest:
                curr_curvature = np.array(r.restKappas())[1:-1, bending_axis]
            else:
                curr_curvature = np.array(r.deformedConfiguration().kappa)[1:-1, bending_axis]
            if not correct_orientation:
                curr_curvature = -1 * curr_curvature[::-1]
            geo_curvatures.extend(curr_curvature)
            curr_edge_lens = None
            if rest:
                curr_edge_lens = r.restLengths()
            else:
                curr_edge_lens = r.deformedConfiguration().len
            if not correct_orientation:
                curr_edge_lens = curr_edge_lens[::-1]
            # The first edge length is dropped
            curr_edge_lens = curr_edge_lens[1:]
            edge_len.extend(curr_edge_lens)

            curr_edge_width = []
            for i in range(r.numEdges()):
                m = r.material(i)
                [a1, a2, a3, a4] = m.crossSectionBoundaryPts
                width = max(la.norm([a1 - a4]), la.norm([a1 - a2]))
                curr_edge_width.append(width)
            if not correct_orientation:
                curr_edge_width = curr_edge_width[::-1]
            # The first edge width is dropped
            curr_edge_width = curr_edge_width[1:]
            edge_width.extend(curr_edge_width)
            if correct_orientation:
                joint_index.append(linkage.segment(seg_index).startJoint)
            else:
                joint_index.append(linkage.segment(seg_index).endJoint)

        if correct_orientation:
            joint_index.append(linkage.segment(seg_index).endJoint)
        else:
            joint_index.append(linkage.segment(seg_index).startJoint)

        geo_angles = [2 * np.arctan(k/2) for k in geo_curvatures]
        all_ribbon_angles.append(geo_angles)
        all_ribbon_edge_len.append(edge_len)
        all_ribbon_widths.append(edge_width)
        all_ribbon_joints.append(joint_index)
        all_segment_index.append(segment_index)
    return all_ribbon_angles, all_ribbon_edge_len, all_ribbon_num_seg, all_ribbon_widths, all_ribbon_joints, all_extra_first_edge, all_segment_index
