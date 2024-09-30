elastic_rods_dir = '../elastic_rods/python/'
weaving_dir = './'
import os
import os.path as osp
import sys; sys.path.append(elastic_rods_dir); sys.path.append(weaving_dir)
import numpy as np, elastic_rods, linkage_vis
import numpy.linalg as la
from bending_validation import suppress_stdout as so
import matplotlib.pyplot as plt
from elastic_rods import EnergyType, InterleavingType

import analysis_helper, ribbon_linkage_helper, mesh_vis, linkage_utils, compute_curve_from_curvature, pipeline_helper, optimization_visualization_helper, structural_analysis, importlib
importlib.reload(analysis_helper)
importlib.reload(ribbon_linkage_helper)
importlib.reload(mesh_vis)
importlib.reload(linkage_utils)
importlib.reload(compute_curve_from_curvature)
importlib.reload(pipeline_helper)
importlib.reload(optimization_visualization_helper)
importlib.reload(structural_analysis)

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

from pipeline_helper import (initialize_linkage, get_normal_deviation, set_joint_vector_field, stage_1_optimization, initialize_stage_2_optimizer, stage_2_optimization, InputOrganizer, write_all_output, set_surface_view_options, get_structure_analysis_view, contact_optimization, get_double_side_view, show_selected_joints, highlight_rod_and_joint, get_max_distance_to_target_surface, get_average_distance_to_target_joint, get_fixed_boundary_joint)

from optimization_visualization_helper import (compute_visualization_data_from_raw_data, get_objective_components_stage1, get_objective_components_stage2, get_objective_components_stage3, set_figure_label_and_limit, Visualization_Setting, plot_objective, plot_objective_stack, plot_ribbon_component_analysis, insert_nan, get_grad_norm_components_stage2, get_grad_norm_components_stage3, plot_merged_ribbon_component_analysis)
import vis.fields
import matplotlib.cm as cm
import time

import matplotlib.pyplot as plt
import json
import pickle
import gzip

# Parallelism settings.
import parallelism
parallelism.set_max_num_tbb_threads(24)
parallelism.set_hessian_assembly_num_threads(8)
parallelism.set_gradient_assembly_num_threads(8)

import multiprocessing as mp
import py_newton_optimizer

# Optimization parameters.
OPTS = py_newton_optimizer.NewtonOptimizerOptions()
OPTS.gradTol = 1e-8
OPTS.verbose = 1;
OPTS.beta = 1e-8
OPTS.niter = 200
OPTS.verboseNonPosDef = False
rw = 0.1
sw = 10
drw = 0.01
dsw = 0.1

# If DEBUG set to true, only compute four data point for visualization for each stage.
DEBUG = False
# If only_two_stage, then the contact optimization is not used. 
only_two_stage = False

def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

vs = Visualization_Setting()

with open(osp.join(weaving_dir + 'woven_model.json')) as f:
    data = json.load(f)

data_root = 'optimization_diagram_results'
compute_visualization_data = True
use_svg = False
def get_optimization_diagram_worker(model_info):
    ''' Run multi-stage optimization given the information about the model. Then for each iteration, compute data about the model and the ribbons. Lastly, generate plots for the objective functions and the descriptive data. 
    '''
    # Load model info from the json data.
    thickness, width, name, use_constant_width, width_scale, scale_joint_weight, update_attraction_weight, number_of_updates, fix_boundary, only_two_stage = model_info['thickness'], model_info['width'], model_info['name'], model_info['constant_cross_section'], model_info['cross_section_scale'], model_info['scale_joint_weight'], model_info['update_attraction_weight'], model_info['number_of_updates'], model_info['fix_boundary'], model_info['only_two_stage']
    
    print('Processing ', name)
    # Load feature joint weight if there is any.
    joint_weight, scale, joint_list = 0, 0, []
    if float(scale_joint_weight.split(', ')[0]) != -1:
        joint_weight, scale, joint_list = float(scale_joint_weight.split(', ')[0]), float(scale_joint_weight.split(', ')[1]), [int(x) for x in scale_joint_weight.split(', ')[2:]]
    
    # Set the interleaving type with special consideration for the bunny.
    interleaving_type = InterleavingType.triaxialWeave
    if name in ['bunny_head_small_triaxial_1', 'owl_1', 'clam_1']:
        interleaving_type = InterleavingType.weaving
    io = InputOrganizer(name, thickness, width, weaving_dir)

    # Create folders for storing computed data.
    if not os.path.exists('{}/{}'.format(data_root, name)):
        os.makedirs('{}/{}'.format(data_root, name))  

    if not os.path.exists('{}/{}/pickle'.format(data_root, name)):
        os.makedirs('{}/{}/pickle'.format(data_root, name)) 

    # Define data file names. 
    data_filename = '{}/{}_{}_data.npy'.format('{}/{}'.format(data_root, name), name, 'full' if not DEBUG else 'finite_sample')
    stage_1_data_filename = '{}/{}_stage_1.npy'.format('{}/{}'.format(data_root, name), name)
    stage_1_pickle_filename = '{}/{}_stage_1.pkl.gz'.format('{}/{}'.format(data_root, name), name)
    stage_2_data_filename = '{}/{}_stage_2.npy'.format('{}/{}'.format(data_root, name), name)
    stage_2_pickle_filename = '{}/{}_stage_2.pkl.gz'.format('{}/{}'.format(data_root, name), name)
    stage_2_weight_change_filename = '{}/{}_stage_2_weight_change.npy'.format('{}/{}'.format(data_root, name), name)
    stage_2_target_weight_filename = '{}/{}_stage_2_target_weight.npy'.format('{}/{}'.format(data_root, name), name)
    stage_3_data_filename = '{}/{}_stage_3.npy'.format('{}/{}'.format(data_root, name), name)
    stage_3_pickle_filename = '{}/{}_stage_3.pkl.gz'.format('{}/{}'.format(data_root, name), name)
    stage_3_weight_filename = '{}/{}_contact_weight_stage_3.npy'.format('{}/{}'.format(data_root, name), name)

    stage_1_vis_data_filename = '{}/{}_stage_1_vis.npy'.format('{}/{}'.format(data_root, name), name)
    stage_2_vis_data_filename = '{}/{}_stage_2_vis.npy'.format('{}/{}'.format(data_root, name), name)
    stage_3_vis_data_filename = '{}/{}_stage_3_vis.npy'.format('{}/{}'.format(data_root, name), name)

    init_time_filename = '{}/{}_init_time.npy'.format('{}/{}'.format(data_root, name), name)

    print('Optimizing ', name)
    # Initialize the linkage class.
    start_time = time.time()
    with so(): curved_linkage = initialize_linkage(surface_path = io.SURFACE_PATH, useCenterline = True, model_path = io.MODEL_PATH, cross_section = io.RIBBON_CS, subdivision_res = io.SUBDIVISION_RESOLUTION, interleaving_type=interleaving_type, use_constant_width = use_constant_width, width_scale = width_scale)
    print( io.SURFACE_PATH, io.MODEL_PATH,  io.RIBBON_CS,  io.SUBDIVISION_RESOLUTION, interleaving_type,  use_constant_width,  width_scale)
    # Fix the twisted ribbon for the bunny.
    if name == 'bunny_head_small_triaxial_1':
        print('Recompute normal for bunny')
        input_joint_normals = np.reshape(curved_linkage.get_closest_point_normal(curved_linkage.jointPositions()), (curved_linkage.numJoints(), 3))
        current_joint_normals = [curved_linkage.joint(i).normal for i in range(curved_linkage.numJoints())]
        current_joint_normals[35] = input_joint_normals[35]
        with so(): curved_linkage = initialize_linkage(surface_path = io.SURFACE_PATH, useCenterline = True, model_path = io.MODEL_PATH, cross_section = io.RIBBON_CS, subdivision_res = io.SUBDIVISION_RESOLUTION, use_constant_width=use_constant_width, interleaving_type=InterleavingType.triaxialWeave, input_joint_normals = current_joint_normals)

    # Set design parameter to include both rest length and rest curvature.
    curved_linkage.set_design_parameter_config(use_restLen = True, use_restKappa = True)
    curved_save_tgt_joint_pos = curved_linkage.jointPositions()
    # For some model the boundary joint positions need to be fixed. 
    fixed_boundary_joints = []
    if fix_boundary:
        fixed_boundary_joints = get_fixed_boundary_joint(curved_linkage)
    # Scale the attraction weights for feature joints if any. 
    if float(scale_joint_weight.split(', ')[0]) != -1:
        curved_linkage.scaleJointWeights(joint_weight, scale, joint_list)

    curved_linkage.attraction_weight = 1e-5
    optimizer = None
    after_initialization_time = time.time()
    np.save(init_time_filename, after_initialization_time - start_time)
    # Run or load stage 1. 
    if os.path.isfile(stage_1_data_filename):
        iterateData = np.load(stage_1_data_filename, allow_pickle = True)
        curved_linkage.setExtendedDoFsPSRL(iterateData[-1]['extendedDoFsPSRL'])
        with so(): elastic_rods.compute_equilibrium(curved_linkage, callback = None, options = OPTS, fixedVars = fixed_boundary_joints)
    # Plot Stage 1 Objective.
    dps_objective_elastic, dps_objective_smooth, dps_objective_length, dps_total_objective, dps_colors, dps_labels = get_objective_components_stage1(iterateData, vs)
    # plot_objective(vs, iterateData, dps_total_objective, '{}/{}_stage_1_objective.{}'.format('{}/{}'.format(data_root, name), name, 'svg' if use_svg else 'png'), vs.stage_1_label)
    # for i in range(len(iterateData)):
    #     plot_objective_stack(vs, iterateData, dps_total_objective, [dps_objective_elastic, np.array(dps_objective_smooth)+np.array(dps_objective_length)], dps_colors, dps_labels, '{}/video_images/{}_stage_1_objective_stack_{:03d}.{}'.format('{}/{}'.format(data_root, name), name, i, 'svg' if use_svg else 'png'), vs.stage_1_label, iteration = i)
    # Run or load stage 2. Need to save or load additional weight change iteration info.
    if os.path.isfile(stage_2_data_filename) and os.path.isfile(stage_2_weight_change_filename):
        opt_iterateData = np.load(stage_2_data_filename, allow_pickle = True)
        weight_change_iters = np.load(stage_2_weight_change_filename, allow_pickle = True)
        curved_linkage.setExtendedDoFsPSRL(opt_iterateData[-1]['extendedDoFsPSRL'])
        with so(): elastic_rods.compute_equilibrium(curved_linkage, callback = None, options = OPTS, fixedVars = fixed_boundary_joints)
    # Plot Stage 2 Objective.
    opt_objective_elastic, opt_objective_target, opt_objective_length, opt_objective_smooth, opt_total_objective, opt_colors, opt_labels, opt_objective_original_smooth = get_objective_components_stage2(opt_iterateData, vs, weight_change_iters)
    # opt_grad_norm_elastic, opt_grad_norm_target, opt_grad_norm_length, opt_grad_norm_smooth, opt_total_grad_norm, opt_colors, opt_labels, opt_grad_norm_original_smooth = get_grad_norm_components_stage2(opt_iterateData, vs, weight_change_iters)
    # plot_objective(vs, opt_iterateData, opt_total_objective, '{}/{}_stage_2_objective.{}'.format('{}/{}'.format(data_root, name), name, 'svg' if use_svg else 'png'), vs.stage_2_label)
    # for i in range(len(opt_iterateData)):
    #     plot_objective_stack(vs, opt_iterateData, opt_total_objective, [opt_objective_elastic, np.array(opt_objective_length)+np.array(opt_objective_smooth), opt_objective_target], opt_colors, opt_labels, '{}/video_images/{}_stage_2_objective_stack_{:03d}.{}'.format('{}/{}'.format(data_root, name), name, i, 'svg' if use_svg else 'png'), vs.stage_2_label, weight_change_iters[1:], iteration = i)
    # plot_objective_stack(vs, opt_iterateData, opt_total_grad_norm, [opt_grad_norm_elastic, np.array(opt_grad_norm_length)+np.array(opt_grad_norm_smooth), opt_grad_norm_target], opt_colors, opt_labels, '{}/{}_stage_2_grad_norm_stack.{}'.format('{}/{}'.format(data_root, name), name, 'svg' if use_svg else 'png'), vs.stage_2_label, weight_change_iters[1:], grad_norm = True)

    # Run or load stage 3.
    if (os.path.isfile(stage_3_data_filename)):
        contact_iterateData = np.load(stage_3_data_filename, allow_pickle = True)
        curved_linkage.setExtendedDoFsPSRL(contact_iterateData[-1]['extendedDoFsPSRL'])
        with so(): elastic_rods.compute_equilibrium(curved_linkage, callback = None, options = OPTS, fixedVars = fixed_boundary_joints)

    # Plot Stage 3 Objective.
    if not only_two_stage:
        contact_objective_elastic, contact_objective_target, contact_objective_length, contact_objective_smooth, contact_total_objective, contact_contact_force, contact_colors, contact_labels = get_objective_components_stage3(contact_iterateData, vs)
        # contact_grad_norm_elastic, contact_grad_norm_target, contact_grad_norm_length, contact_grad_norm_smooth, contact_total_grad_norm, contact_grad_norm_contact, contact_colors, contact_labels = get_grad_norm_components_stage3(contact_iterateData, vs)
        # plot_objective(vs, contact_iterateData, contact_total_objective, '{}/{}_stage_3_objective.{}'.format('{}/{}'.format(data_root, name), name, 'svg' if use_svg else 'png'), vs.stage_3_label)
        # for i in range(len(contact_iterateData)):
        #     plot_objective_stack(vs, contact_iterateData, contact_total_objective, [contact_objective_elastic, np.array(contact_objective_length)+np.array(contact_objective_smooth), contact_objective_target, contact_contact_force], contact_colors, contact_labels, '{}/video_images/{}_stage_3_objective_stack_{:03d}.{}'.format('{}/{}'.format(data_root, name), name, i, 'svg' if use_svg else 'png'), vs.stage_3_label, iteration = i)
        # plot_objective_stack(vs, contact_iterateData, contact_total_grad_norm, [contact_grad_norm_elastic, np.array(contact_grad_norm_length)+np.array(contact_grad_norm_smooth), contact_grad_norm_target, contact_grad_norm_contact], contact_colors, contact_labels, '{}/{}_stage_3_grad_norm_stack.{}'.format('{}/{}'.format(data_root, name), name, 'svg' if use_svg else 'png'), vs.stage_3_label, grad_norm = True)
    else:
        contact_objective_elastic, contact_objective_target, contact_objective_length, contact_objective_smooth, contact_total_objective, contact_contact_force, contact_colors, contact_labels = [], [], [], [], [], [], [], []

    if not compute_visualization_data:
        return 
    print('Computing Visualization Data ', name)        
    # Need to normalize the smoothing term to stage 2 and stage 3.
    combined_weighted_smoothing = np.concatenate((np.array(dps_objective_smooth) * sw / dsw, opt_objective_smooth, contact_objective_smooth), axis = None)
    # Compute and plot detailed description of the ribbons.
        # Stage 1
    if os.path.isfile(stage_1_vis_data_filename):
        dps_total_absolute_curvature, dps_total_ribbon_length, dps_distance_to_surface, dps_distance_to_joint, dps_eqm_dofs, dps_elastic_energy, dps_separation_force, dps_tangential_force = np.load(stage_1_vis_data_filename, allow_pickle = True)
    else:
        dps_total_absolute_curvature, dps_total_ribbon_length, dps_distance_to_surface, dps_distance_to_joint, dps_eqm_dofs, dps_elastic_energy, dps_separation_force, dps_tangential_force = compute_visualization_data_from_raw_data(iterateData, io, '{}/{}'.format(data_root, name), 1, DEBUG, interleaving_type=interleaving_type, use_constant_width = use_constant_width, width_scale = width_scale, is_bunny = False, fixed_boundary_joints = fixed_boundary_joints)
        stage_1_vis_data = np.array([dps_total_absolute_curvature, dps_total_ribbon_length, dps_distance_to_surface, dps_distance_to_joint, dps_eqm_dofs, dps_elastic_energy, dps_separation_force, dps_tangential_force])
        np.save(stage_1_vis_data_filename, stage_1_vis_data)

        # Stage 2
    if os.path.isfile(stage_2_vis_data_filename):
        opt_total_absolute_curvature, opt_total_ribbon_length, opt_distance_to_surface, opt_distance_to_joint, opt_eqm_dofs, opt_elastic_energy, opt_separation_force, opt_tangential_force = np.load(stage_2_vis_data_filename, allow_pickle = True)
    else:
        opt_total_absolute_curvature, opt_total_ribbon_length, opt_distance_to_surface, opt_distance_to_joint, opt_eqm_dofs, opt_elastic_energy, opt_separation_force, opt_tangential_force  = compute_visualization_data_from_raw_data(opt_iterateData, io, '{}/{}'.format(data_root, name), 2, DEBUG, interleaving_type=interleaving_type, use_constant_width = use_constant_width, width_scale = width_scale, is_bunny = False, weight_change_iters = weight_change_iters, fixed_boundary_joints = fixed_boundary_joints)
        stage_2_vis_data = np.array([opt_total_absolute_curvature, opt_total_ribbon_length, opt_distance_to_surface, opt_distance_to_joint, opt_eqm_dofs, opt_elastic_energy, opt_separation_force, opt_tangential_force])

        np.save(stage_2_vis_data_filename, stage_2_vis_data)
    # Insert np.nan to separate the weight change 
    # [opt_total_absolute_curvature, opt_total_ribbon_length, opt_distance_to_surface, opt_distance_to_joint, opt_eqm_dofs, opt_elastic_energy, opt_separation_force, opt_tangential_force] = insert_nan([opt_total_absolute_curvature, opt_total_ribbon_length, opt_distance_to_surface, opt_distance_to_joint, opt_eqm_dofs, opt_elastic_energy, opt_separation_force, opt_tangential_force], weight_change_iters[1:-1])

        # Stage 3
    if os.path.isfile(stage_3_vis_data_filename):
        contact_total_absolute_curvature, contact_total_ribbon_length, contact_distance_to_surface, contact_distance_to_joint, contact_eqm_dofs, contact_elastic_energy, contact_separation_force, contact_tangential_force = np.load(stage_3_vis_data_filename, allow_pickle = True)
    elif not only_two_stage:
        contact_total_absolute_curvature, contact_total_ribbon_length, contact_distance_to_surface, contact_distance_to_joint, contact_eqm_dofs, contact_elastic_energy, contact_separation_force, contact_tangential_force  = compute_visualization_data_from_raw_data(contact_iterateData, io, '{}/{}'.format(data_root, name), 3, DEBUG, interleaving_type=interleaving_type, use_constant_width = use_constant_width, width_scale = width_scale, is_bunny = False, fixed_boundary_joints = fixed_boundary_joints)
        stage_3_vis_data = np.array([contact_total_absolute_curvature, contact_total_ribbon_length, contact_distance_to_surface, contact_distance_to_joint, contact_eqm_dofs, contact_elastic_energy, contact_separation_force, contact_tangential_force])
        np.save(stage_3_vis_data_filename, stage_3_vis_data)

    else:
        contact_total_absolute_curvature, contact_total_ribbon_length, contact_distance_to_surface, contact_distance_to_joint, contact_eqm_dofs, contact_elastic_energy, contact_separation_force, contact_tangential_force = [], [], [], [], [], [], [], []

    # Aggregate data across three stages to set the limits in the plots and show continuity from stages to stages.
    combined_energy = np.concatenate((dps_elastic_energy, opt_elastic_energy, contact_elastic_energy), axis = None)
    combined_rest_length = np.concatenate((dps_total_ribbon_length, opt_total_ribbon_length, contact_total_ribbon_length), axis = None)
    combined_total_absolute_curvature = np.concatenate((dps_total_absolute_curvature, opt_total_absolute_curvature, contact_total_absolute_curvature), axis = None)
    combined_distance_to_surface = np.concatenate((dps_distance_to_surface, opt_distance_to_surface, contact_distance_to_surface), axis = None)
    combined_dis_to_target_joint = np.concatenate((dps_distance_to_joint, opt_distance_to_joint, contact_distance_to_joint), axis = None)
    combined_separation_force = np.concatenate((dps_separation_force, opt_separation_force, contact_separation_force), axis = None)
    combined_tangential_force = np.concatenate((dps_tangential_force, opt_tangential_force, contact_tangential_force), axis = None)

    # # Save the aggregated data. (Unused because the data for each stage is saved separately.)
    # curr_vis_data = np.array([combined_energy, combined_rest_length, combined_total_absolute_curvature, combined_distance_to_surface, combined_dis_to_target_joint, combined_separation_force, combined_tangential_force, dps_eqm_dofs, opt_eqm_dofs, contact_eqm_dofs])
    # np.save(data_filename, curr_vis_data)
    
    # # Don't show forces data in stage 1.
    # combined_separation_force_clipped = np.concatenate((np.ones(len(iterateData)) * combined_separation_force[len(iterateData)], combined_separation_force[len(iterateData):]), axis = None)
    # combined_tangential_force_clipped = np.concatenate((np.ones(len(iterateData)) * combined_tangential_force[len(iterateData)], combined_tangential_force[len(iterateData):]), axis = None)

    # Plot the computed information for each stage.
    plot_ribbon_component_analysis(iterateData, dps_elastic_energy, dps_total_ribbon_length, np.array(dps_objective_smooth) * sw / dsw, dps_total_absolute_curvature, dps_distance_to_surface, dps_distance_to_joint, dps_separation_force, dps_tangential_force, DEBUG, combined_energy, combined_rest_length, combined_weighted_smoothing, combined_total_absolute_curvature, combined_distance_to_surface, combined_dis_to_target_joint, combined_separation_force, combined_tangential_force, vs, name, 1, vs.stage_1_label)
    plot_ribbon_component_analysis(opt_iterateData, opt_elastic_energy, opt_total_ribbon_length, opt_objective_original_smooth, opt_total_absolute_curvature, opt_distance_to_surface, opt_distance_to_joint, opt_separation_force, opt_tangential_force, DEBUG, combined_energy, combined_rest_length, combined_weighted_smoothing, combined_total_absolute_curvature, combined_distance_to_surface, combined_dis_to_target_joint, combined_separation_force, combined_tangential_force, vs, name, 2, vs.stage_2_label)
    plot_ribbon_component_analysis(contact_iterateData, contact_elastic_energy, contact_total_ribbon_length, contact_objective_smooth, contact_total_absolute_curvature, contact_distance_to_surface, contact_distance_to_joint, contact_separation_force, contact_tangential_force, DEBUG, combined_energy, combined_rest_length, combined_weighted_smoothing, combined_total_absolute_curvature, combined_distance_to_surface, combined_dis_to_target_joint, combined_separation_force, combined_tangential_force, vs, name, 3, vs.stage_3_label)
    for i in range(len(iterateData)):
        plot_ribbon_component_analysis(iterateData, dps_elastic_energy, dps_total_ribbon_length, np.array(dps_objective_smooth) * sw / dsw, dps_total_absolute_curvature, dps_distance_to_surface, dps_distance_to_joint, dps_separation_force, dps_tangential_force, DEBUG, combined_energy, combined_rest_length, combined_weighted_smoothing, combined_total_absolute_curvature, combined_distance_to_surface, combined_dis_to_target_joint, combined_separation_force, combined_tangential_force, vs, name, 1, vs.stage_1_label, iteration = i)
    for i in range(len(opt_iterateData)):
        plot_ribbon_component_analysis(opt_iterateData, opt_elastic_energy, opt_total_ribbon_length, opt_objective_original_smooth, opt_total_absolute_curvature, opt_distance_to_surface, opt_distance_to_joint, opt_separation_force, opt_tangential_force, DEBUG, combined_energy, combined_rest_length, combined_weighted_smoothing, combined_total_absolute_curvature, combined_distance_to_surface, combined_dis_to_target_joint, combined_separation_force, combined_tangential_force, vs, name, 2, vs.stage_2_label, iteration = i)
    for i in range(len(contact_iterateData)):
        plot_ribbon_component_analysis(contact_iterateData, contact_elastic_energy, contact_total_ribbon_length, contact_objective_smooth, contact_total_absolute_curvature, contact_distance_to_surface, contact_distance_to_joint, contact_separation_force, contact_tangential_force, DEBUG, combined_energy, combined_rest_length, combined_weighted_smoothing, combined_total_absolute_curvature, combined_distance_to_surface, combined_dis_to_target_joint, combined_separation_force, combined_tangential_force, vs, name, 3, vs.stage_3_label, iteration = i)
get_optimization_diagram_worker(data['models'][1])
# print(len(data['models']))
# get_optimization_diagram_worker(data['models'][2])
# NUM_CORE = 4
# pool = mp.Pool(NUM_CORE)
# pool.map(get_optimization_diagram_worker, (model_info for model_info in data['models'][1:]))


# for index, model_info in enumerate(data['models']):
#     if index not in [0, 1, 2, 3, 4, 5, 11, 12, 13]:
#         get_optimization_diagram_worker(model_info)


