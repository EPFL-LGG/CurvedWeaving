elastic_rods_dir = '../elastic_rods/python/'
weaving_dir = './'
import os
import os.path as osp
import sys; sys.path.append(elastic_rods_dir); sys.path.append(weaving_dir)

import numpy as np, elastic_rods
from bending_validation import suppress_stdout as so

import analysis_helper, ribbon_linkage_helper, mesh_vis, linkage_utils, compute_curve_from_curvature, pipeline_helper, optimization_visualization_helper, structural_analysis, importlib
importlib.reload(analysis_helper)
importlib.reload(ribbon_linkage_helper)
importlib.reload(mesh_vis)
importlib.reload(linkage_utils)
importlib.reload(compute_curve_from_curvature)
importlib.reload(pipeline_helper)
importlib.reload(optimization_visualization_helper)
importlib.reload(structural_analysis)


from pipeline_helper import (stage_1_optimization, initialize_stage_2_optimizer_gh, initialize_stage_2_optimizer, stage_2_optimization, InputOrganizer, contact_optimization, get_fixed_boundary_joint, set_design_parameters_from_topology)

from optimization_visualization_helper import (compute_visualization_data_from_raw_data, get_objective_components_stage1, get_objective_components_stage2, get_objective_components_stage3, Visualization_Setting, plot_objective, plot_objective_stack, plot_ribbon_component_analysis, get_grad_norm_components_stage2, get_grad_norm_components_stage3)
import time

import matplotlib.pyplot as plt
import pickle
import gzip

# Parallelism settings.
import parallelism
parallelism.set_max_num_tbb_threads(8)
parallelism.set_hessian_assembly_num_threads(8)
parallelism.set_gradient_assembly_num_threads(8)

import py_newton_optimizer

from gh_optimizer_helper import WeavingIO

# Optimization parameters.
OPTS = py_newton_optimizer.NewtonOptimizerOptions()
OPTS.gradTol = 1e-8
OPTS.verbose = 1;
OPTS.beta = 1e-8
OPTS.niter = 200
OPTS.verboseNonPosDef = False
rw = 0.1
sw = 10
drw = 0.001
dsw = 0.01

# If DEBUG set to true, only compute four data point for visualization for each stage.
DEBUG = False

vs = Visualization_Setting()


data_root = 'grasshopper/optimization_diagram_results'
compute_visualization_data = False
use_svg = False

def optimize_model(input_model_name):
    ''' Run multi-stage optimization given the information about the model. Then for each iteration, compute data about the model and the ribbons. Lastly, generate plots for the objective functions and the descriptive data. 
    '''
    # Load model info from the json data.
    model_name = str(input_model_name)
    filename = osp.join(weaving_dir + 'grasshopper/inputs/{}.json'.format(input_model_name))
    
    ################################################
    ########### INIT. LINKAGE
    ################################################
    weavingIO = WeavingIO(filename=filename)
    if weavingIO == None:
        return False
    curved_linkage = weavingIO.surface_attracted_linkage
    if curved_linkage == None:
        return False
    
    io = InputOrganizer(model_name, weavingIO.ref_height, weavingIO.ref_width, weaving_dir)
    io.SUBDIVISION_RESOLUTION = weavingIO.subdivision
    interleaving_type = weavingIO.interleaving
    update_attraction_weight = weavingIO.update_attraction_weight
    print('Processing model:', model_name)
        
    # TODO: Load feature joint weight if there is any.
    scale_joint_weight = "-1"
    joint_weight, scale, joint_list = 0, 0, []
    if float(scale_joint_weight.split(', ')[0]) != -1:
        joint_weight, scale, joint_list = float(scale_joint_weight.split(', ')[0]), float(scale_joint_weight.split(', ')[1]), [int(x) for x in scale_joint_weight.split(', ')[2:]]
    
    # Create folders for storing computed data.
    if not os.path.exists('{}/{}'.format(data_root, model_name)):
        os.makedirs('{}/{}'.format(data_root, model_name))  

    if not os.path.exists('{}/{}/pickle'.format(data_root, model_name)):
        os.makedirs('{}/{}/pickle'.format(data_root, model_name)) 

    # Define data file names. 
    data_filename = '{}/{}_{}_data.npy'.format('{}/{}'.format(data_root, model_name), model_name, 'full' if not DEBUG else 'finite_sample')
    stage_1_data_filename = '{}/{}_stage_1.npy'.format('{}/{}'.format(data_root, model_name), model_name)
    stage_1_pickle_filename = '{}/{}_stage_1.pkl.gz'.format('{}/{}'.format(data_root, model_name), model_name)
    stage_2_data_filename = '{}/{}_stage_2.npy'.format('{}/{}'.format(data_root, model_name), model_name)
    stage_2_pickle_filename = '{}/{}_stage_2.pkl.gz'.format('{}/{}'.format(data_root, model_name), model_name)
    stage_2_weight_change_filename = '{}/{}_stage_2_weight_change.npy'.format('{}/{}'.format(data_root, model_name), model_name)
    stage_2_target_weight_filename = '{}/{}_stage_2_target_weight.npy'.format('{}/{}'.format(data_root, model_name), model_name)
    stage_3_data_filename = '{}/{}_stage_3.npy'.format('{}/{}'.format(data_root, model_name), model_name)
    stage_3_pickle_filename = '{}/{}_stage_3.pkl.gz'.format('{}/{}'.format(data_root, model_name), model_name)
    stage_3_weight_filename = '{}/{}_contact_weight_stage_3.npy'.format('{}/{}'.format(data_root, model_name), model_name)

    stage_1_vis_data_filename = '{}/{}_stage_1_vis.npy'.format('{}/{}'.format(data_root, model_name), model_name)
    stage_2_vis_data_filename = '{}/{}_stage_2_vis.npy'.format('{}/{}'.format(data_root, model_name), model_name)
    stage_3_vis_data_filename = '{}/{}_stage_3_vis.npy'.format('{}/{}'.format(data_root, model_name), model_name)

    stage_1_dof_filename = '{}/{}_stage_1_dof.npy'.format('{}/{}'.format(data_root, model_name), model_name)
    stage_2_dof_filename = '{}/{}_stage_2_dof.npy'.format('{}/{}'.format(data_root, model_name), model_name)
    stage_3_dof_filename = '{}/{}_stage_3_dof.npy'.format('{}/{}'.format(data_root, model_name), model_name)

    stage_2_solverStatus_filename = '{}/{}_stage_2_solverStatus.npy'.format('{}/{}'.format(data_root, model_name), model_name)
    stage_3_solverStatus_filename = '{}/{}_stage_3_solverStatus.npy'.format('{}/{}'.format(data_root, model_name), model_name)

    init_time_filename = '{}/{}_init_time.npy'.format('{}/{}'.format(data_root, model_name), model_name)

    print('Optimizing ', model_name)
    
    # Initialize the linkage class.
    start_time = time.time()

    # Initialize design parameters.
    set_design_parameters_from_topology(curved_linkage, io)

    # Set design parameter to include both rest length and rest curvature.
    curved_linkage.set_design_parameter_config(use_restLen = True, use_restKappa = True)
    curved_save_tgt_joint_pos = curved_linkage.jointPositions()
    
    # For some model the boundary joint positions need to be fixed. 
    fixed_boundary_joints = curved_linkage.fixed_vars
            
    # Scale the attraction weights for feature joints if any. 
    if float(scale_joint_weight.split(', ')[0]) != -1:
        curved_linkage.scaleJointWeights(joint_weight, scale, joint_list)

    curved_linkage.attraction_weight = 1e-5
    optimizer = None
    after_initialization_time = time.time()
    np.save(init_time_filename, after_initialization_time - start_time)

    contact_objective_elastic, contact_objective_target, contact_objective_length, contact_objective_smooth, contact_total_objective, contact_contact_force, contact_colors, contact_labels = [], [], [], [], [], [], [], []
    #############################################################################
    # STAGE 1
    # Run or load stage 1. 
    #############################################################################
    if os.path.isfile(stage_1_data_filename):
        iterateData = np.load(stage_1_data_filename, allow_pickle = True)
        curved_linkage.setExtendedDoFsPSRL(np.load(stage_1_dof_filename))
        #with so(): 
        elastic_rods.compute_equilibrium(curved_linkage, callback = None, options = OPTS, fixedVars = fixed_boundary_joints)
    else:
        print('Stage 1 ', model_name)
        iterateData, _ = stage_1_optimization(curved_linkage, drw, dsw, None)
        if not compute_visualization_data:
            for i in range(len(iterateData)):
                iterateData[i]['extendedDoFsPSRL'] = []
        np.save(stage_1_data_filename, iterateData)
        np.save(stage_1_dof_filename, curved_linkage.getExtendedDoFsPSRL())
        pickle.dump(curved_linkage, gzip.open(stage_1_pickle_filename, 'w'))
        # with so(): 
        elastic_rods.compute_equilibrium(curved_linkage, callback = None, options = OPTS, fixedVars = fixed_boundary_joints)

    # Plot Stage 1 Objective.
    dps_objective_elastic, dps_objective_smooth, dps_objective_length, dps_total_objective, dps_colors, dps_labels = get_objective_components_stage1(iterateData, vs)
    plot_objective(vs, iterateData, dps_total_objective, '{}/{}_stage_1_objective.{}'.format('{}/{}'.format(data_root, model_name), model_name, 'svg' if use_svg else 'png'), vs.stage_1_label)
    plot_objective_stack(vs, iterateData, dps_total_objective, [dps_objective_elastic, np.array(dps_objective_smooth)+np.array(dps_objective_length)], dps_colors, dps_labels, '{}/{}_stage_1_objective_stack.{}'.format('{}/{}'.format(data_root, model_name), model_name, 'svg' if use_svg else 'png'), vs.stage_1_label)
    
    
    #############################################################################
    # STAGE 2
    # Run or load stage 2. Need to save or load additional weight change iteration info.
    #############################################################################
    if weavingIO.num_optimization_stages == 2 or weavingIO.num_optimization_stages == 3:
        if os.path.isfile(stage_2_data_filename) and os.path.isfile(stage_2_weight_change_filename):
            opt_iterateData = np.load(stage_2_data_filename, allow_pickle = True)
            weight_change_iters = np.load(stage_2_weight_change_filename, allow_pickle = True)
            curved_linkage.setExtendedDoFsPSRL(np.load(stage_2_dof_filename))
            with so(): elastic_rods.compute_equilibrium(curved_linkage, callback = None, options = OPTS, fixedVars = fixed_boundary_joints)
        else:
            print("Stage 2 ", model_name)
            optimizer = initialize_stage_2_optimizer_gh(curved_linkage, weavingIO.target_vertices, weavingIO.target_faces, curved_save_tgt_joint_pos, None, rw, sw, fixed_boundary_joint = fixed_boundary_joints, inner_gradTol = 1e-6) 
            if float(scale_joint_weight.split(', ')[0]) != -1:
                optimizer.scaleJointWeights(joint_weight, scale, joint_list)
            optimizer, opt_iterateData, weight_change_iters, target_weight, solverStatusList = stage_2_optimization(optimizer, curved_linkage, io.SURFACE_PATH, curved_save_tgt_joint_pos, None, -1, fixed_boundary_joints, update_attraction_weight, weavingIO.number_of_updates, True)
            # assert 0
            if not compute_visualization_data:
                for i in range(len(opt_iterateData)):
                    opt_iterateData[i]['extendedDoFsPSRL'] = []
            np.save(stage_2_data_filename, opt_iterateData)
            np.save(stage_2_weight_change_filename, weight_change_iters)
            np.save(stage_2_target_weight_filename, target_weight)
            np.save(stage_2_dof_filename, curved_linkage.getExtendedDoFsPSRL())
            np.save(stage_2_solverStatus_filename, solverStatusList)
            pickle.dump(curved_linkage, gzip.open(stage_2_pickle_filename, 'w'))
            
        # Plot Stage 2 Objective.
        opt_objective_elastic, opt_objective_target, opt_objective_length, opt_objective_smooth, opt_total_objective, opt_colors, opt_labels, opt_objective_original_smooth = get_objective_components_stage2(opt_iterateData, vs, weight_change_iters)
        opt_grad_norm_elastic, opt_grad_norm_target, opt_grad_norm_length, opt_grad_norm_smooth, opt_total_grad_norm, opt_colors, opt_labels, opt_grad_norm_original_smooth = get_grad_norm_components_stage2(opt_iterateData, vs, weight_change_iters)
        plot_objective(vs, opt_iterateData, opt_total_objective, '{}/{}_stage_2_objective.{}'.format('{}/{}'.format(data_root, model_name), model_name, 'svg' if use_svg else 'png'), vs.stage_2_label)
        plot_objective_stack(vs, opt_iterateData, opt_total_objective, [opt_objective_elastic, np.array(opt_objective_length)+np.array(opt_objective_smooth), opt_objective_target], opt_colors, opt_labels, '{}/{}_stage_2_objective_stack.{}'.format('{}/{}'.format(data_root, model_name), model_name, 'svg' if use_svg else 'png'), vs.stage_2_label, weight_change_iters[1:])
        plot_objective_stack(vs, opt_iterateData, opt_total_grad_norm, [opt_grad_norm_elastic, np.array(opt_grad_norm_length)+np.array(opt_grad_norm_smooth), opt_grad_norm_target], opt_colors, opt_labels, '{}/{}_stage_2_grad_norm_stack.{}'.format('{}/{}'.format(data_root, model_name), model_name, 'svg' if use_svg else 'png'), vs.stage_2_label, weight_change_iters[1:], grad_norm = True)

    #############################################################################
    # STAGE 3
    # Run or load stage 3.
    #############################################################################
    if weavingIO.num_optimization_stages == 3:
        if (os.path.isfile(stage_3_data_filename)):
            contact_iterateData = np.load(stage_3_data_filename, allow_pickle = True)
            curved_linkage.setExtendedDoFsPSRL(np.load(stage_3_dof_filename))
            with so(): elastic_rods.compute_equilibrium(curved_linkage, callback = None, options = OPTS, fixedVars = fixed_boundary_joints)
        else:
            print('Stage 3 ', model_name)
            optimizer = initialize_stage_2_optimizer(curved_linkage,  weavingIO.target_vertices, weavingIO.target_faces, curved_save_tgt_joint_pos, None, rw, sw, fixed_boundary_joint = fixed_boundary_joints, inner_gradTol = 1e-7)
            if float(scale_joint_weight.split(', ')[0]) != -1:
                optimizer.scaleJointWeights(joint_weight, scale, joint_list)
            from structural_analysis import weavingCrossingForceMagnitudes
            cfm = weavingCrossingForceMagnitudes(curved_linkage, True)
            normalActivationThreshold = np.min(np.percentile(cfm[:, 0], 75), 0)
            print("Normal Activation Threshold: ", normalActivationThreshold)
            optimizer, contact_iterateData, solverStatus = contact_optimization(optimizer, curved_linkage, None, minRestLen=-1, contact_weight = 5e6, normalActivationThreshold = normalActivationThreshold, normalWeight = 10, tangentialWeight = 1, torqueWeight = 0, maxIter=2000, update_attraction_weight = update_attraction_weight, callback_freq = 1)
            if not compute_visualization_data:
                for i in range(len(contact_iterateData)):
                    contact_iterateData[i]['extendedDoFsPSRL'] = []
            np.save(stage_3_weight_filename, [optimizer.objective.terms[0].term.weight, optimizer.objective.terms[1].term.weight, optimizer.objective.terms[2].term.weight, optimizer.objective.terms[3].term.weight, optimizer.objective.terms[-1].term.weight])
            np.save(stage_3_data_filename, contact_iterateData)
            np.save(stage_3_dof_filename, curved_linkage.getExtendedDoFsPSRL())
            np.save(stage_3_solverStatus_filename, solverStatus)
            pickle.dump(curved_linkage, gzip.open(stage_3_pickle_filename, 'w'))
     
        
        # Plot Stage 3 Objective.
        contact_objective_elastic, contact_objective_target, contact_objective_length, contact_objective_smooth, contact_total_objective, contact_contact_force, contact_colors, contact_labels = get_objective_components_stage3(contact_iterateData, vs)
        contact_grad_norm_elastic, contact_grad_norm_target, contact_grad_norm_length, contact_grad_norm_smooth, contact_total_grad_norm, contact_grad_norm_contact, contact_colors, contact_labels = get_grad_norm_components_stage3(contact_iterateData, vs)
        plot_objective(vs, contact_iterateData, contact_total_objective, '{}/{}_stage_3_objective.{}'.format('{}/{}'.format(data_root, model_name), model_name, 'svg' if use_svg else 'png'), vs.stage_3_label)
        plot_objective_stack(vs, contact_iterateData, contact_total_objective, [contact_objective_elastic, np.array(contact_objective_length)+np.array(contact_objective_smooth), contact_objective_target, contact_contact_force], contact_colors, contact_labels, '{}/{}_stage_3_objective_stack.{}'.format('{}/{}'.format(data_root, model_name), model_name, 'svg' if use_svg else 'png'), vs.stage_3_label)
        plot_objective_stack(vs, contact_iterateData, contact_total_grad_norm, [contact_grad_norm_elastic, np.array(contact_grad_norm_length)+np.array(contact_grad_norm_smooth), contact_grad_norm_target, contact_grad_norm_contact], contact_colors, contact_labels, '{}/{}_stage_3_grad_norm_stack.{}'.format('{}/{}'.format(data_root, model_name), model_name, 'svg' if use_svg else 'png'), vs.stage_3_label, grad_norm = True)

    #############################################################################
    # Compute visualization data
    #############################################################################
    if not compute_visualization_data:
        return True
    
    print('Computing Visualization Data ', model_name)    
    contact_total_absolute_curvature, contact_total_ribbon_length, contact_distance_to_surface, contact_distance_to_joint, contact_eqm_dofs, contact_elastic_energy, contact_separation_force, contact_tangential_force = [], [], [], [], [], [], [], []
                
    # Need to normalize the smoothing term to stage 2 and stage 3.
    combined_weighted_smoothing = np.concatenate((np.array(dps_objective_smooth) * sw / dsw, opt_objective_smooth, contact_objective_smooth), axis = None)
    # Compute and plot detailed description of the ribbons.
    # Stage 1
    if os.path.isfile(stage_1_vis_data_filename):
        dps_total_absolute_curvature, dps_total_ribbon_length, dps_distance_to_surface, dps_distance_to_joint, dps_eqm_dofs, dps_elastic_energy, dps_separation_force, dps_tangential_force = np.load(stage_1_vis_data_filename, allow_pickle = True)
    else:
        dps_total_absolute_curvature, dps_total_ribbon_length, dps_distance_to_surface, dps_distance_to_joint, dps_eqm_dofs, dps_elastic_energy, dps_separation_force, dps_tangential_force = compute_visualization_data_from_raw_data(iterateData, io, '{}/{}'.format(data_root, model_name), 1, DEBUG, interleaving_type=interleaving_type, use_constant_width = use_constant_width, width_scale = width_scale, is_bunny = False, fixed_boundary_joints = fixed_boundary_joints)
        stage_1_vis_data = np.array([dps_total_absolute_curvature, dps_total_ribbon_length, dps_distance_to_surface, dps_distance_to_joint, dps_eqm_dofs, dps_elastic_energy, dps_separation_force, dps_tangential_force])
        np.save(stage_1_vis_data_filename, stage_1_vis_data)

    # Stage 2
    if weavingIO.num_optimization_stages == 2 or weavingIO.num_optimization_stages == 3:
        if os.path.isfile(stage_2_vis_data_filename):
            opt_total_absolute_curvature, opt_total_ribbon_length, opt_distance_to_surface, opt_distance_to_joint, opt_eqm_dofs, opt_elastic_energy, opt_separation_force, opt_tangential_force = np.load(stage_2_vis_data_filename, allow_pickle = True)
        else:
            opt_total_absolute_curvature, opt_total_ribbon_length, opt_distance_to_surface, opt_distance_to_joint, opt_eqm_dofs, opt_elastic_energy, opt_separation_force, opt_tangential_force  = compute_visualization_data_from_raw_data(opt_iterateData, io, '{}/{}'.format(data_root, model_name), 2, DEBUG, interleaving_type=interleaving_type, use_constant_width = use_constant_width, width_scale = width_scale, is_bunny = False, weight_change_iters = weight_change_iters, fixed_boundary_joints = fixed_boundary_joints)
            stage_2_vis_data = np.array([opt_total_absolute_curvature, opt_total_ribbon_length, opt_distance_to_surface, opt_distance_to_joint, opt_eqm_dofs, opt_elastic_energy, opt_separation_force, opt_tangential_force])
            np.save(stage_2_vis_data_filename, stage_2_vis_data)
        # Insert np.nan to separate the weight change 
        # [opt_total_absolute_curvature, opt_total_ribbon_length, opt_distance_to_surface, opt_distance_to_joint, opt_eqm_dofs, opt_elastic_energy, opt_separation_force, opt_tangential_force] = insert_nan([opt_total_absolute_curvature, opt_total_ribbon_length, opt_distance_to_surface, opt_distance_to_joint, opt_eqm_dofs, opt_elastic_energy, opt_separation_force, opt_tangential_force], weight_change_iters[1:-1])

        # Stage 3
    if weavingIO.num_optimization_stages == 3:
        if os.path.isfile(stage_3_vis_data_filename):
            contact_total_absolute_curvature, contact_total_ribbon_length, contact_distance_to_surface, contact_distance_to_joint, contact_eqm_dofs, contact_elastic_energy, contact_separation_force, contact_tangential_force = np.load(stage_3_vis_data_filename, allow_pickle = True)
        else:
            contact_total_absolute_curvature, contact_total_ribbon_length, contact_distance_to_surface, contact_distance_to_joint, contact_eqm_dofs, contact_elastic_energy, contact_separation_force, contact_tangential_force  = compute_visualization_data_from_raw_data(contact_iterateData, io, '{}/{}'.format(data_root, model_name), 3, DEBUG, interleaving_type=interleaving_type, use_constant_width = use_constant_width, width_scale = width_scale, is_bunny = False, fixed_boundary_joints = fixed_boundary_joints)
            stage_3_vis_data = np.array([contact_total_absolute_curvature, contact_total_ribbon_length, contact_distance_to_surface, contact_distance_to_joint, contact_eqm_dofs, contact_elastic_energy, contact_separation_force, contact_tangential_force])
            np.save(stage_3_vis_data_filename, stage_3_vis_data)

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
    plot_ribbon_component_analysis(iterateData, dps_elastic_energy, dps_total_ribbon_length, np.array(dps_objective_smooth) * sw / dsw, dps_total_absolute_curvature, dps_distance_to_surface, dps_distance_to_joint, dps_separation_force, dps_tangential_force, DEBUG, combined_energy, combined_rest_length, combined_weighted_smoothing, combined_total_absolute_curvature, combined_distance_to_surface, combined_dis_to_target_joint, combined_separation_force, combined_tangential_force, vs, model_name, 1, vs.stage_1_label)
    plot_ribbon_component_analysis(opt_iterateData, opt_elastic_energy, opt_total_ribbon_length, opt_objective_original_smooth, opt_total_absolute_curvature, opt_distance_to_surface, opt_distance_to_joint, opt_separation_force, opt_tangential_force, DEBUG, combined_energy, combined_rest_length, combined_weighted_smoothing, combined_total_absolute_curvature, combined_distance_to_surface, combined_dis_to_target_joint, combined_separation_force, combined_tangential_force, vs, model_name, 2, vs.stage_2_label)

    if weavingIO.num_optimization_stages == 3:
        plot_ribbon_component_analysis(contact_iterateData, contact_elastic_energy, contact_total_ribbon_length, contact_objective_smooth, contact_total_absolute_curvature, contact_distance_to_surface, contact_distance_to_joint, contact_separation_force, contact_tangential_force, DEBUG, combined_energy, combined_rest_length, combined_weighted_smoothing, combined_total_absolute_curvature, combined_distance_to_surface, combined_dis_to_target_joint, combined_separation_force, combined_tangential_force, vs, model_name, 3, vs.stage_3_label)
    
    return True