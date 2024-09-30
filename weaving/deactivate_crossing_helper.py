elastic_rods_dir = '../../elastic_rods/python/'
weaving_dir = '../'
import os
import os.path as osp
import sys; sys.path.append(elastic_rods_dir); sys.path.append(weaving_dir)
import numpy as np, elastic_rods, linkage_vis
import numpy.linalg as la
from bending_validation import suppress_stdout as so
import matplotlib.pyplot as plt
from elastic_rods import EnergyType, InterleavingType


def write_deactivated_linkage(original_model_name, deactive_model_name, deactive_index, deactive_vertex_map, nbs):
    with open(original_model_name, 'r') as linkage_file, open(deactive_model_name, 'w') as output_file:
        content = linkage_file.readlines()
        point_list = []
        edge_list = []
        for line in content:
            if 'v ' in line:
                point = np.array([float(x) for x in line.split(' ')[1:]])
                point_list.append(point)
            if 'l ' in line:
                edge_list.append([int(x) for x in line.split(' ')[1:]])
        
        for pt_idx, pt in enumerate(point_list):
            if pt_idx != deactive_index:
                output_file.write('v {} {} {}\n'.format(pt[0], pt[1], pt[2]))
        
        for edge in edge_list:
            if deactive_index + 1 not in edge:
    #             The plus one here is due to obj index from 1
                output_file.write('l {} {}\n'.format(deactive_vertex_map[edge[0]-1] + 1, deactive_vertex_map[edge[1]-1]+1))
            
        output_file.write('l {} {}\n'.format(deactive_vertex_map[nbs[0]]+1, deactive_vertex_map[nbs[2]]+1))
        output_file.write('l {} {}\n'.format(deactive_vertex_map[nbs[1]]+1, deactive_vertex_map[nbs[3]]+1))

def invalid_segment(linkage, seg_idx):
    return seg_idx > linkage.numSegments()

def get_neighbors(linkage, joint_index):
    ju = linkage.joint(joint_index)
    neighbors = [None, None, None, None]
    if not invalid_segment(linkage, ju.segments_A[0]):
        neighbors[0] = linkage.segment(ju.segments_A[0]).endJoint if linkage.segment(ju.segments_A[0]).startJoint == joint_index else linkage.segment(ju.segments_A[0]).startJoint

    if not invalid_segment(linkage, ju.segments_B[0]):
        neighbors[1] = linkage.segment(ju.segments_B[0]).endJoint if linkage.segment(ju.segments_B[0]).startJoint == joint_index else linkage.segment(ju.segments_B[0]).startJoint

    if not invalid_segment(linkage, ju.segments_A[1]):
        neighbors[2] = linkage.segment(ju.segments_A[1]).endJoint if linkage.segment(ju.segments_A[1]).startJoint == joint_index else linkage.segment(ju.segments_A[1]).startJoint

    if not invalid_segment(linkage, ju.segments_B[1]):
        neighbors[3] = linkage.segment(ju.segments_B[1]).endJoint if linkage.segment(ju.segments_B[1]).startJoint == joint_index else linkage.segment(ju.segments_B[1]).startJoint
    return neighbors

def copy_over_under(deactivate_curved_linkage, curved_linkage, deactive_vertex_map, deactive_index):
    def get_opposite_joint_type(curr_type):
        if curr_type == curved_linkage.joint(0).Type.A_OVER_B:
            return curved_linkage.joint(0).Type.B_OVER_A
        return curved_linkage.joint(0).Type.A_OVER_B

    for i in range(curved_linkage.numJoints()):
        if i != deactive_index:
            deactivate_ju = deactivate_curved_linkage.joint(deactive_vertex_map[i])
            ju = curved_linkage.joint(i)
            deactivate_nbs = get_neighbors(deactivate_curved_linkage, deactive_vertex_map[i])
            origin_nbs = [deactive_vertex_map[j] if j != None else None for j in get_neighbors(curved_linkage, i)]
    #         Need to check whether the labels for A B segments are flipped and whether the normal for the joints are flipped
            if (set([origin_nbs[0], origin_nbs[2]]) == set([deactivate_nbs[0], deactivate_nbs[2]]) or set([origin_nbs[1], origin_nbs[3]]) == set([deactivate_nbs[1], deactivate_nbs[3]])) == (np.dot(ju.normal, deactivate_curved_linkage.joint(deactive_vertex_map[i]).normal) > 0):
                deactivate_curved_linkage.joint(deactive_vertex_map[i]).type = curved_linkage.joint(i).type
            else:
                deactivate_curved_linkage.joint(deactive_vertex_map[i]).type = get_opposite_joint_type(curved_linkage.joint(i).type)

def get_segment_index_from_joints(linkage, start_joint, end_joint):

    ju = linkage.joint(start_joint)
    curr_nbs = ju.neighbors()
    if end_joint == curr_nbs[0]:
        return (ju.segments_A[0])
    elif end_joint == curr_nbs[1]:
        if invalid_segment(linkage, ju.segments_B[0]) and invalid_segment(linkage, ju.segments_A[1]):
            return None
        elif invalid_segment(linkage, ju.segments_B[0]):
            return ju.segments_A[1]
        else:
            return ju.segments_B[0]
    elif end_joint == curr_nbs[2]:
        if invalid_segment(linkage, ju.segments_A[1]):
            if invalid_segment(linkage, ju.segments_B[1]):
                return None
            return ju.segments_B[1]
        else:
            return ju.segments_A[1]
    elif end_joint == curr_nbs[3]:
        if invalid_segment(linkage, ju.segments_B[1]):
            return None
        return ju.segments_B[1]

def get_angle_from_kappa(kappa):
    return 2 * np.arctan(kappa/2)
def get_kappa_from_angle(angle):
    return 2 * np.tan(angle/2)


def no_resample_deactivation(deactivate_curved_linkage, curved_linkage, deactive_vertex_map, nbs, deactive_index, io):
    deactivate_design_parameter = deactivate_curved_linkage.getDesignParameters()
    design_parameter = curved_linkage.getDesignParameters()
    deactivate_rl_design_parameter_offset = deactivate_curved_linkage.numRestKappaVars()
    rl_design_parameter_offset = curved_linkage.numRestKappaVars()
    segment_rk_offset = []
    for i in range(deactivate_curved_linkage.numSegments()):
        segment_rk_offset.append(deactivate_curved_linkage.segment(i).rod.numRestKappaVars())
    segment_rk_offset = [sum(segment_rk_offset[:i]) for i in range(len(segment_rk_offset)+1)]
    for k in range(2):
        deactivate_seg_idx = get_segment_index_from_joints(deactivate_curved_linkage, deactive_vertex_map[nbs[0+k]], deactive_vertex_map[nbs[2+k]])
    #     Among these two adjacent segments, only one of the two neighbor joints is a start joint since the segments are sorted per rod.
        original_start_joint = nbs[0+k] if curved_linkage.segment(get_segment_index_from_joints(curved_linkage, deactive_index, nbs[0+k])).startJoint == nbs[0+k] else nbs[2+k]
        original_end_joint = nbs[0+k] if original_start_joint != nbs[0+k] else nbs[2+k]
        origin_seg_idx_1 = get_segment_index_from_joints(curved_linkage, original_start_joint, deactive_index)
        origin_seg_idx_2 = get_segment_index_from_joints(curved_linkage, original_end_joint, deactive_index)

    #     Set rest length
        deactivate_design_parameter[deactivate_rl_design_parameter_offset + deactivate_seg_idx] = design_parameter[rl_design_parameter_offset + origin_seg_idx_1] + design_parameter[rl_design_parameter_offset + origin_seg_idx_2]
        
    #     Set rest kappa
        curvature_1 = design_parameter[origin_seg_idx_1 * (io.SUBDIVISION_RESOLUTION - 1): (origin_seg_idx_1+1)* (io.SUBDIVISION_RESOLUTION - 1)]
        curvature_2 = design_parameter[origin_seg_idx_2 * (io.SUBDIVISION_RESOLUTION - 1): (origin_seg_idx_2+1)* (io.SUBDIVISION_RESOLUTION - 1)]
        accumulated_kappa = np.concatenate((curvature_1, curvature_2), axis=None)
        downsampled_kappa = np.array([get_kappa_from_angle(get_angle_from_kappa(accumulated_kappa[2 * i]) + get_angle_from_kappa(accumulated_kappa[2 * i+1])) for i in range(len(curvature_1))])
        if deactive_vertex_map[original_start_joint] != deactivate_curved_linkage.segment(deactivate_seg_idx).startJoint:
            downsampled_kappa *= -1
            downsampled_kappa = downsampled_kappa[::-1]
        deactivate_design_parameter[segment_rk_offset[deactivate_seg_idx]: segment_rk_offset[deactivate_seg_idx+1]] = downsampled_kappa

    for i in range(curved_linkage.numSegments()):
        seg = curved_linkage.segment(i)
        start_joint, end_joint = seg.startJoint, seg.endJoint
        if deactive_index not in [start_joint, end_joint]:
            deactivate_seg_idx = get_segment_index_from_joints(deactivate_curved_linkage, deactive_vertex_map[start_joint], deactive_vertex_map[end_joint])
            if deactivate_seg_idx == None:
                print(start_joint, end_joint)
            else:
                deactivate_design_parameter[deactivate_rl_design_parameter_offset + deactivate_seg_idx] = design_parameter[rl_design_parameter_offset + i]
                deactivate_design_parameter[segment_rk_offset[deactivate_seg_idx]: segment_rk_offset[(deactivate_seg_idx+1)]] = design_parameter[i * (io.SUBDIVISION_RESOLUTION - 1): (i+1)* (io.SUBDIVISION_RESOLUTION - 1)]
    deactivate_curved_linkage.setDesignParameters(deactivate_design_parameter)
    return deactivate_design_parameter

def build_design_parameter_offset(deactivate_curved_linkage, curved_linkage):
    deactivate_rl_design_parameter_offset = deactivate_curved_linkage.numRestKappaVars()
    rl_design_parameter_offset = curved_linkage.numRestKappaVars()
    deactivate_segment_rk_offset = []
    for i in range(deactivate_curved_linkage.numSegments()):
        deactivate_segment_rk_offset.append(deactivate_curved_linkage.segment(i).rod.numRestKappaVars())
    deactivate_segment_rk_offset = [sum(deactivate_segment_rk_offset[:i]) for i in range(len(deactivate_segment_rk_offset)+1)]
    segment_rk_offset = []
    for i in range(curved_linkage.numSegments()):
        segment_rk_offset.append(curved_linkage.segment(i).rod.numRestKappaVars())
    segment_rk_offset = [sum(segment_rk_offset[:i]) for i in range(len(segment_rk_offset)+1)]
    return deactivate_rl_design_parameter_offset, rl_design_parameter_offset, deactivate_segment_rk_offset, segment_rk_offset

def check_curvature_flipped(curved_linkage, deactivate_curved_linkage, original_start_joint, deactivate_start_joint, deactive_vertex_map):
    # This check only works when the surface is orientable and the directions of the segment align with the start joint's normal
    original_normal = curved_linkage.joint(original_start_joint).normal 
    deactivate_normal = deactivate_curved_linkage.joint(deactivate_start_joint).normal
    curvature_flipped = (np.dot(original_normal, deactivate_normal) > 0) != (deactive_vertex_map[original_start_joint] == deactivate_start_joint)
    return curvature_flipped

def resample_deactivation(deactivate_curved_linkage, curved_linkage, deactive_vertex_map, nbs, deactive_index, io):
    jpos = np.reshape(deactivate_curved_linkage.jointPositions(), (deactivate_curved_linkage.numJoints(), 3))

    # Resample rod
    for k in range(2):
        deactivate_seg_idx = get_segment_index_from_joints(deactivate_curved_linkage, deactive_vertex_map[nbs[0+k]], deactive_vertex_map[nbs[2+k]])
        print(deactivate_seg_idx)
    #     Among these two adjacent segments, only one of the two neighbor joints is a start joint since the segments are sorted per rod.
        original_start_joint = nbs[0+k] if curved_linkage.segment(get_segment_index_from_joints(curved_linkage, deactive_index, nbs[0+k])).startJoint == nbs[0+k] else nbs[2+k]
        original_end_joint = nbs[0+k] if original_start_joint != nbs[0+k] else nbs[2+k]
        origin_seg_idx_1 = get_segment_index_from_joints(curved_linkage, original_start_joint, deactive_index)
        origin_seg_idx_2 = get_segment_index_from_joints(curved_linkage, original_end_joint, deactive_index)
        #     Resamples edges in rod
        new_start_joint, new_end_joint = deactivate_curved_linkage.segment(deactivate_seg_idx).startJoint, deactivate_curved_linkage.segment(deactivate_seg_idx).endJoint
        new_rod_seg = elastic_rods.RodLinkage.RodSegment(jpos[new_start_joint], jpos[new_end_joint], io.SUBDIVISION_RESOLUTION * 2 - 1)
        new_rod_seg.startJoint = new_start_joint
        new_rod_seg.endJoint = new_end_joint
        deactivate_curved_linkage._set_segment(new_rod_seg, deactivate_seg_idx)
    segmentRestLenGuess = []
    for i in range(deactivate_curved_linkage.numSegments()):
        seg = deactivate_curved_linkage.segment(i)
        segmentRestLenGuess.append(la.norm(jpos[seg.startJoint] - jpos[seg.endJoint]))
    deactivate_curved_linkage.constructSegmentRestLenToEdgeRestLenMapTranspose(segmentRestLenGuess)

    # Update hidden variables in the linkage
    for i in range(deactivate_curved_linkage.numSegments()):
        deactivate_curved_linkage.segment(i).setMinimalTwistThetas()
    deactivate_curved_linkage.updateSourceFrame()
    deactivate_curved_linkage.setMaterial(elastic_rods.RodMaterial('rectangle', 2000, 0.3, io.RIBBON_CS, stiffAxis=elastic_rods.StiffAxis.D1))

    # Because the segments are resampled, the design parameter length changed; need to construct from scratch. 
    deactivate_design_parameter = np.zeros(deactivate_curved_linkage.numRestKappaVars() + deactivate_curved_linkage.numSegments())
    design_parameter = curved_linkage.getDesignParameters()

    deactivate_rl_design_parameter_offset, rl_design_parameter_offset, deactivate_segment_rk_offset, segment_rk_offset = build_design_parameter_offset(deactivate_curved_linkage, curved_linkage)
    # Compute Design Parameters
    #   Resampled rods
    for k in range(2):
        deactivate_seg_idx = get_segment_index_from_joints(deactivate_curved_linkage, deactive_vertex_map[nbs[0+k]], deactive_vertex_map[nbs[2+k]])
    #     Among these two adjacent segments, only one of the two neighbor joints is a start joint since the segments are sorted per rod.
        original_start_joint = nbs[0+k] if curved_linkage.segment(get_segment_index_from_joints(curved_linkage, deactive_index, nbs[0+k])).startJoint == nbs[0+k] else nbs[2+k]
        original_end_joint = nbs[0+k] if original_start_joint != nbs[0+k] else nbs[2+k]
        origin_seg_idx_1 = get_segment_index_from_joints(curved_linkage, original_start_joint, deactive_index)
        origin_seg_idx_2 = get_segment_index_from_joints(curved_linkage, original_end_joint, deactive_index)

    #     Set rest length
        deactivate_design_parameter[deactivate_rl_design_parameter_offset + deactivate_seg_idx] = design_parameter[rl_design_parameter_offset + origin_seg_idx_1] + design_parameter[rl_design_parameter_offset + origin_seg_idx_2]
        
    #     Set rest kappa
        curvature_1 = design_parameter[segment_rk_offset[origin_seg_idx_1]: segment_rk_offset[origin_seg_idx_1 + 1]]
        curvature_2 = design_parameter[segment_rk_offset[origin_seg_idx_2]: segment_rk_offset[origin_seg_idx_2 + 1]]
        accumulated_kappa = np.concatenate((curvature_1, curvature_2), axis=None)
        if check_curvature_flipped(curved_linkage, deactivate_curved_linkage, original_start_joint, deactivate_curved_linkage.segment(deactivate_seg_idx).startJoint, deactive_vertex_map):
            accumulated_kappa *= -1
            accumulated_kappa = accumulated_kappa[::-1]
        deactivate_design_parameter[deactivate_segment_rk_offset[deactivate_seg_idx]: deactivate_segment_rk_offset[deactivate_seg_idx+1]] = accumulated_kappa

    #   Original rods
    for i in range(curved_linkage.numSegments()):
        seg = curved_linkage.segment(i)
        start_joint, end_joint = seg.startJoint, seg.endJoint
        if deactive_index not in [start_joint, end_joint]:
            # Rest length
            deactivate_seg_idx = get_segment_index_from_joints(deactivate_curved_linkage, deactive_vertex_map[start_joint], deactive_vertex_map[end_joint])
            deactivate_design_parameter[deactivate_rl_design_parameter_offset + deactivate_seg_idx] = design_parameter[rl_design_parameter_offset + i]
            # Rest kappa
            orignal_curvature = design_parameter[segment_rk_offset[i]: segment_rk_offset[i + 1]]
            if check_curvature_flipped(curved_linkage, deactivate_curved_linkage, start_joint, deactivate_curved_linkage.segment(deactivate_seg_idx).startJoint, deactive_vertex_map):
                orignal_curvature *= -1
                orignal_curvature = orignal_curvature[::-1]
            deactivate_design_parameter[deactivate_segment_rk_offset[deactivate_seg_idx]: deactivate_segment_rk_offset[(deactivate_seg_idx+1)]] = orignal_curvature

    deactivate_curved_linkage.setDesignParameters(deactivate_design_parameter)
    return deactivate_design_parameter

























