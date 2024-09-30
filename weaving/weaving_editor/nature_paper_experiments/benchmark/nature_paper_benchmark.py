#!/usr/bin/env python
elastic_rods_dir = '../../../../elastic_rods/python/'
weaving_dir = '../../../'
import os
import os.path as osp
import sys; sys.path.append(elastic_rods_dir); sys.path.append(weaving_dir)
import numpy as np, elastic_rods, linkage_vis
import numpy.linalg as la
from bending_validation import suppress_stdout as so
import matplotlib.pyplot as plt
from elastic_rods import EnergyType, InterleavingType

# weaving
import analysis_helper, ribbon_linkage_helper, mesh_vis, linkage_utils, compute_curve_from_curvature, pipeline_helper, importlib
importlib.reload(analysis_helper)
importlib.reload(ribbon_linkage_helper)
importlib.reload(mesh_vis)
importlib.reload(linkage_utils)
importlib.reload(compute_curve_from_curvature)
importlib.reload(pipeline_helper)
from analysis_helper import (compare_turning_angle,
                            is_on_sphere, 
                            get_distance_to_center_scalar_field, 
                            plot_curvatures, 
                            get_curvature_scalar_field,
                            construct_elastic_rod_loop_from_rod_segments, 
                            concatenate_rod_properties_from_rod_segments, 
                            compute_min_distance_rigid_transformation)
from ribbon_linkage_helper import (update_rest_curvature, 
                                   set_ribbon_linkage,
                                   export_linkage_geometry_to_obj,
                                   write_linkage_ribbon_output_florin)

from compute_curve_from_curvature import (match_geo_curvature_and_edge_len, get_all_curve_pattern)
from linkage_utils import order_segments_by_ribbons, get_turning_angle_and_length_from_ordered_rods

from pipeline_helper import (initialize_linkage, get_normal_deviation, set_joint_vector_field, stage_1_optimization, initialize_stage_2_optimizer, stage_2_optimization, InputOrganizer, write_all_output, set_surface_view_options, get_structure_analysis_view, get_max_distance_to_target_surface, Visualization_Setting, set_figure_label_and_limit, contact_optimization)

import vis.fields
import matplotlib.cm as cm
import time
import json
import time
import benchmark
import io as python_io
import parallelism
parallelism.set_max_num_tbb_threads(12)
parallelism.set_hessian_assembly_num_threads(4)
parallelism.set_gradient_assembly_num_threads(4)

OPTS = elastic_rods.NewtonOptimizerOptions()
OPTS.gradTol = 1e-6
OPTS.verbose = 10;
OPTS.beta = 1e-8
OPTS.niter = 100
OPTS.verboseNonPosDef = False
rw = 0.01
sw = 0.01


with open(osp.join(weaving_dir + 'woven_model.json')) as f:
    data = json.load(f)

stats = []
for model_info in data['models']:
    benchmark.reset()
    current_stats = {}
    thickness, width, name, use_constant_width, width_scale, scale_joint_weight, update_attraction_weight, number_of_updates, fix_boundary = model_info['thickness'], model_info['width'], model_info['name'], model_info['constant_cross_section'], model_info['cross_section_scale'], model_info['scale_joint_weight'], model_info['update_attraction_weight'], model_info['number_of_updates'], model_info['fix_boundary']
    
    current_stats['Display Name'] = model_info['display_name']
    joint_weight, scale, joint_list = 0, 0, []
    if float(scale_joint_weight.split(', ')[0]) != -1:
        joint_weight, scale, joint_list = float(scale_joint_weight.split(', ')[0]), float(scale_joint_weight.split(', ')[1]), [int(x) for x in scale_joint_weight.split(', ')[2:]]
        
    interleaving_type = InterleavingType.triaxialWeave
    if name == 'kleinbottle_projected_1':
        interleaving_type = InterleavingType.weaving
    io = InputOrganizer(name, thickness, width, weaving_dir)
    
    before_initialization_time = time.time()
    with so(): curved_linkage = initialize_linkage(surface_path = io.SURFACE_PATH, useCenterline = True, model_path = io.MODEL_PATH, cross_section = io.RIBBON_CS, subdivision_res = io.SUBDIVISION_RESOLUTION, interleaving_type=interleaving_type, use_constant_width = use_constant_width, width_scale = width_scale)
    curved_linkage.set_design_parameter_config(use_restLen = True, use_restKappa = True)
    curved_save_tgt_joint_pos = curved_linkage.jointPositions()
    after_initialization_time = time.time()

    
    before_stage_1_time = time.time()
    iterateData = stage_1_optimization(curved_linkage, rw, sw, None)
    after_stage_1_time = time.time()
    
    fixed_boundary_joints = []
    if fix_boundary:
        fixed_boundary_joints = get_fixed_boundary_joint(curved_linkage)
        
    if float(scale_joint_weight.split(', ')[0]) != -1:
        curved_linkage.scaleJointWeights(joint_weight, scale, joint_list)
    with so(): elastic_rods.compute_equilibrium(curved_linkage, callback = None, options = OPTS, fixedVars = fixed_boundary_joints)
    
    before_stage_2_time = time.time()
    optimizer = initialize_stage_2_optimizer(curved_linkage, io.SURFACE_PATH, curved_save_tgt_joint_pos, None, fixed_boundary_joint = fixed_boundary_joints)
    if float(scale_joint_weight.split(', ')[0]) != -1:
        optimizer.scaleJointWeights(joint_weight, scale, joint_list)
    with so(): optimizer, opt_iterateData = stage_2_optimization(optimizer, curved_linkage, io.SURFACE_PATH, curved_save_tgt_joint_pos, None, -1, update_attraction_weight, number_of_updates)
    after_stage_2_time = time.time()
    
    curved_optimizer_energy = curved_linkage.energy()
    validation_curved_linkage = optimizer.getLinesearchWeaverLinkage()
    validation_curved_linkage.attraction_weight = 1e-7
    with so(): elastic_rods.compute_equilibrium(validation_curved_linkage, options = OPTS, fixedVars = fixed_boundary_joints)
    validation_curved_view = linkage_vis.LinkageViewer(validation_curved_linkage, width=1024, height=640)
    validation_curved_energy = validation_curved_linkage.energy()
    validation_error = abs((validation_curved_energy-curved_optimizer_energy)/curved_optimizer_energy)
    
    deviation, _, _, _ = get_normal_deviation(curved_linkage)
    
    old_stdout = sys.stdout
    new_stdout = python_io.StringIO()
    sys.stdout = new_stdout
    benchmark.report()
    sys.stdout = old_stdout
    
    np.save('dofs/benchmark_{}_dof.npy'.format(name), curved_linkage.getExtendedDoFsPSRL())
    current_stats['Number of Crossings'] = curved_linkage.numJoints()
    current_stats['Number of Ribbon Segments'] = curved_linkage.numSegments()
    current_stats['DoF'] = curved_linkage.numExtendedDoFPSRL()
    current_stats['Initialization'] = after_initialization_time - before_initialization_time
    current_stats['Stage 1'] = after_stage_1_time - before_stage_1_time
    current_stats['Stage 2'] = after_stage_2_time - before_stage_2_time
    current_stats['Validation Error'] = validation_error
    current_stats['Norm. Max Distance to Surface'] = get_max_distance_to_target_surface(curved_linkage)
    current_stats['Max Normal Deviation'] = np.max(deviation)
    current_stats['detail_benchmark'] = new_stdout.getvalue()
    stats.append(current_stats)
    with open('benchmark_results.json', 'w') as f:
        json.dump(current_stats, f, indent = 4)





