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

# weaving
import analysis_helper, ribbon_linkage_helper, mesh_vis, linkage_utils, compute_curve_from_curvature, importlib
importlib.reload(analysis_helper)
importlib.reload(ribbon_linkage_helper)
importlib.reload(mesh_vis)
importlib.reload(linkage_utils)
importlib.reload(compute_curve_from_curvature)
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
                                   write_linkage_ribbon_output_florin,
                                   write_distance_to_linkage_mesh)

from compute_curve_from_curvature import (match_geo_curvature_and_edge_len, get_all_curve_pattern)
from linkage_utils import order_segments_by_ribbons, get_turning_angle_and_length_from_ordered_rods
import pickle 
import gzip

import vis.fields
import matplotlib.cm as cm
import time

import cross_section_scaling
importlib.reload(cross_section_scaling)

import linkage_optimization
import structural_analysis

from pipeline_helper import (initialize_linkage, get_normal_deviation, set_joint_vector_field, stage_1_optimization, initialize_stage_2_optimizer, stage_2_optimization, InputOrganizer, write_all_output, set_surface_view_options, get_structure_analysis_view, contact_optimization, get_double_side_view, show_selected_joints, highlight_rod_and_joint, get_max_distance_to_target_surface, get_average_distance_to_target_surface, get_average_distance_to_target_joint)

import multiprocessing as mp
from structural_analysis import weavingCrossingForceMagnitudes
import py_newton_optimizer
import copy

data_root = 'optimization_diagram_results'
use_svg = False
def insert_nan(list_of_data, weight_change_iters):
    print("Weight change iterations: ", weight_change_iters)
    def helper(data, pos):
        return np.insert(data, pos, np.nan)
    # for iter_num in weight_change_iters:
    #     for i in range(len(list_of_data)):
    #         list_of_data[i] = helper(list_of_data[i], iter_num)
    return list_of_data

def compute_visualization_data_from_raw_data(iterateData, io, file_path, stage_index, DEBUG = True, interleaving_type=InterleavingType.weaving, use_constant_width = True, width_scale = [1, 1], is_bunny = False, weight_change_iters = [0], fixed_boundary_joints = []):
    if is_bunny:
        with so(): visualization_linkage = initialize_linkage(surface_path = io.SURFACE_PATH, useCenterline = True, model_path = io.MODEL_PATH, cross_section = io.RIBBON_CS, subdivision_res = io.SUBDIVISION_RESOLUTION, use_constant_width=False, interleaving_type=InterleavingType.weaving)
        print('Recompute normal for bunny')
        input_joint_normals = np.reshape(visualization_linkage.get_closest_point_normal(visualization_linkage.jointPositions()), (visualization_linkage.numJoints(), 3))
        current_joint_normals = [visualization_linkage.joint(i).normal for i in range(visualization_linkage.numJoints())]
        current_joint_normals[35] = input_joint_normals[35]
        with so(): visualization_linkage = initialize_linkage(surface_path = io.SURFACE_PATH, useCenterline = True, model_path = io.MODEL_PATH, cross_section = io.RIBBON_CS, subdivision_res = io.SUBDIVISION_RESOLUTION, interleaving_type=InterleavingType.triaxialWeave, use_constant_width = use_constant_width, width_scale = width_scale, input_joint_normals = current_joint_normals)
    else:
        with so(): visualization_linkage = initialize_linkage(surface_path = io.SURFACE_PATH, useCenterline = True, model_path = io.MODEL_PATH, cross_section = io.RIBBON_CS, subdivision_res = io.SUBDIVISION_RESOLUTION,  interleaving_type=interleaving_type, use_constant_width = use_constant_width, width_scale = width_scale)

    visualization_linkage.set_design_parameter_config(use_restLen = True, use_restKappa = True)
    visualization_linkage.attraction_weight = 1e-5
    curved_save_tgt_joint_pos = visualization_linkage.jointPositions();

    elastic_energy = []
    total_absolute_curvature = []
    total_ribbon_length = []
    distance_to_surface = []
    distance_to_joint = []
    eqm_dofs = []
    separation_force = []
    tangential_force = []
    rk_offset = visualization_linkage.numDoF()
    rl_offset = rk_offset + (visualization_linkage.numRestKappaVars() if visualization_linkage.get_design_parameter_config().restKappa else 0)
    rl_end = rl_offset + (visualization_linkage.numSegments() if visualization_linkage.get_design_parameter_config().restLen else 0)

    OPTS = py_newton_optimizer.NewtonOptimizerOptions()
    OPTS.gradTol = 1e-8
    OPTS.verbose = 1;
    OPTS.beta = 1e-8
    OPTS.niter = 200
    OPTS.verboseNonPosDef = False

    start_time = time.time()

    iter_idx_range = range(len(iterateData)) if not DEBUG else [0, int(len(iterateData)/4), int(3 * len(iterateData)/4), int(len(iterateData)-1)]
    for iter_idx in iter_idx_range:
        pickle_file_name = '{}/pickle/{}_{}.pkl.gz'.format(file_path, stage_index, iter_idx)
        print(iter_idx)
        total_absolute_curvature.append(np.sum(np.abs(iterateData[iter_idx]['extendedDoFsPSRL'][rk_offset:rl_offset])))
        total_ribbon_length.append(np.sum(np.abs(iterateData[iter_idx]['extendedDoFsPSRL'][rl_offset:rl_end])))

        if os.path.isfile('{}/pickle/{}_{}.pkl.gz'.format(file_path, stage_index, iter_idx)):
            visualization_linkage = pickle.load(gzip.open(pickle_file_name, 'r'))
        else:
            visualization_linkage.setExtendedDoFsPSRL(iterateData[iter_idx]['extendedDoFsPSRL'])
            with so(): elastic_rods.compute_equilibrium(visualization_linkage, options = OPTS, fixedVars = fixed_boundary_joints)
        distance_to_surface.append(get_average_distance_to_target_surface(visualization_linkage))
        distance_to_joint.append(get_average_distance_to_target_joint(visualization_linkage, curved_save_tgt_joint_pos))
    #         curr_dof = visualization_linkage.getExtendedDoFsPSRL()
        eqm_dofs.append(visualization_linkage.getExtendedDoFsPSRL())
        elastic_energy.append(visualization_linkage.energy())
        cfm = weavingCrossingForceMagnitudes(visualization_linkage, True)
        separation_force.append(max(max(cfm[:, 0]), 0))
        tangential_force.append(max(cfm[:, 1]))
        pickle.dump(visualization_linkage, gzip.open(pickle_file_name, 'w'))
        
    end_time = time.time()
    print(end_time - start_time)

    return total_absolute_curvature, total_ribbon_length, distance_to_surface, distance_to_joint, eqm_dofs, elastic_energy, separation_force, tangential_force

def get_objective_components_stage1(iterateData, vs):
    dps_objective_elastic = []
    dps_objective_smooth = []
    dps_objective_length = []

    for iter_idx in range(len(iterateData)):
        dps_objective_elastic.append(iterateData[iter_idx]['weighted_energy'])
        dps_objective_smooth.append(iterateData[iter_idx]['weighted_smoothness'])
        dps_objective_length.append(iterateData[iter_idx]['weighted_length'])

    dps_total_objective =  np.array([dps_objective_elastic, dps_objective_smooth, dps_objective_length]).sum(axis=0)
    colors = [vs.elastic_color, vs.regularization_color]
    labels = [vs.elastic_label, vs.regularization_label]
    return dps_objective_elastic, dps_objective_smooth, dps_objective_length, dps_total_objective, colors, labels

def get_objective_components_stage2(iterateData, vs, weight_change_iters):
    opt_objective_elastic = []
    opt_objective_target = []
    opt_objective_length = []
    opt_objective_smooth = []

    for iter_idx in range(len(iterateData)):
        opt_objective_elastic.append(iterateData[iter_idx]['ElasticEnergy'])
        opt_objective_target.append(iterateData[iter_idx]['TargetFitting'])
        opt_objective_length.append(iterateData[iter_idx]['RestLengthMinimization'])
        opt_objective_smooth.append(iterateData[iter_idx]['RestCurvatureSmoothing'])
        
    opt_objective_elastic = np.array(opt_objective_elastic)
    opt_objective_target = np.array(opt_objective_target)
    opt_objective_length = np.array(opt_objective_length)
    opt_objective_smooth = np.array(opt_objective_smooth)
    opt_objective_original_smooth = copy.deepcopy(opt_objective_smooth)

    opt_total_objective =  np.array([opt_objective_elastic, opt_objective_target, opt_objective_length, opt_objective_smooth]).sum(axis=0)
    [opt_objective_elastic, opt_objective_target, opt_objective_length, opt_objective_smooth, opt_total_objective] = insert_nan([opt_objective_elastic, opt_objective_target, opt_objective_length, opt_objective_smooth, opt_total_objective], weight_change_iters[1:])
    
    colors = [vs.elastic_color, vs.regularization_color, vs.target_color]
    labels = [vs.elastic_label, vs.regularization_label, vs.target_label]
    return opt_objective_elastic, opt_objective_target, opt_objective_length, opt_objective_smooth, opt_total_objective, colors, labels, opt_objective_original_smooth

def get_grad_norm_components_stage2(iterateData, vs, weight_change_iters):
    opt_grad_norm_elastic = []
    opt_grad_norm_target = []
    opt_grad_norm_length = []
    opt_grad_norm_smooth = []

    for iter_idx in range(len(iterateData)):
        opt_grad_norm_elastic.append(iterateData[iter_idx]['ElasticEnergy_grad_norm'])
        opt_grad_norm_target.append(iterateData[iter_idx]['TargetFitting_grad_norm'])
        opt_grad_norm_length.append(iterateData[iter_idx]['RestLengthMinimization_grad_norm'])
        opt_grad_norm_smooth.append(iterateData[iter_idx]['RestCurvatureSmoothing_grad_norm'])
        
    opt_grad_norm_elastic = np.array(opt_grad_norm_elastic)
    opt_grad_norm_target = np.array(opt_grad_norm_target)
    opt_grad_norm_length = np.array(opt_grad_norm_length)
    opt_grad_norm_smooth = np.array(opt_grad_norm_smooth)
    opt_grad_norm_original_smooth = copy.deepcopy(opt_grad_norm_smooth)

    opt_total_grad_norm =  np.array([opt_grad_norm_elastic, opt_grad_norm_target, opt_grad_norm_length, opt_grad_norm_smooth]).sum(axis=0)
    [opt_grad_norm_elastic, opt_grad_norm_target, opt_grad_norm_length, opt_grad_norm_smooth, opt_total_grad_norm] = insert_nan([opt_grad_norm_elastic, opt_grad_norm_target, opt_grad_norm_length, opt_grad_norm_smooth, opt_total_grad_norm], weight_change_iters[1:])
    
    opt_colors = [vs.elastic_color, vs.regularization_color, vs.target_color]
    opt_labels = [vs.elastic_label, vs.regularization_label, vs.target_label]
    return opt_grad_norm_elastic, opt_grad_norm_target, opt_grad_norm_length, opt_grad_norm_smooth, opt_total_grad_norm, opt_colors, opt_labels, opt_grad_norm_original_smooth

def get_objective_components_stage3(iterateData, vs):
    contact_objective_elastic = []
    contact_objective_target = []
    contact_objective_length = []
    contact_objective_smooth = []
    contact_objective_contact = []

    for iter_idx in range(len(iterateData)):
        contact_objective_elastic.append(iterateData[iter_idx]['ElasticEnergy'])
        contact_objective_target.append(iterateData[iter_idx]['TargetFitting'])
        contact_objective_length.append(iterateData[iter_idx]['RestLengthMinimization'])
        contact_objective_smooth.append(iterateData[iter_idx]['RestCurvatureSmoothing'])
        contact_objective_contact.append(iterateData[iter_idx]['ContactForce'])
        
    contact_objective_elastic = np.array(contact_objective_elastic)
    contact_objective_target = np.array(contact_objective_target)
    contact_objective_length = np.array(contact_objective_length)
    contact_objective_smooth = np.array(contact_objective_smooth)
    contact_objective_contact = np.array(contact_objective_contact)

    contact_total_objective =  np.array([contact_objective_elastic, contact_objective_target, contact_objective_length, contact_objective_smooth, contact_objective_contact]).sum(axis=0)
    colors = [vs.elastic_color, vs.regularization_color, vs.target_color, vs.contact_color]
    labels = [vs.elastic_label, vs.regularization_label, vs.target_label, vs.contact_label]
    return contact_objective_elastic, contact_objective_target, contact_objective_length, contact_objective_smooth, contact_total_objective, contact_objective_contact, colors, labels

def get_grad_norm_components_stage3(iterateData, vs):
    contact_grad_norm_elastic = []
    contact_grad_norm_target = []
    contact_grad_norm_length = []
    contact_grad_norm_smooth = []
    contact_grad_norm_contact = []

    for iter_idx in range(len(iterateData)):
        contact_grad_norm_elastic.append(iterateData[iter_idx]['ElasticEnergy_grad_norm'])
        contact_grad_norm_target.append(iterateData[iter_idx]['TargetFitting_grad_norm'])
        contact_grad_norm_length.append(iterateData[iter_idx]['RestLengthMinimization_grad_norm'])
        contact_grad_norm_smooth.append(iterateData[iter_idx]['RestCurvatureSmoothing_grad_norm'])
        contact_grad_norm_contact.append(iterateData[iter_idx]['ContactForce_grad_norm'])
        
    contact_grad_norm_elastic = np.array(contact_grad_norm_elastic)
    contact_grad_norm_target = np.array(contact_grad_norm_target)
    contact_grad_norm_length = np.array(contact_grad_norm_length)
    contact_grad_norm_smooth = np.array(contact_grad_norm_smooth)
    contact_grad_norm_contact = np.array(contact_grad_norm_contact)

    contact_total_grad_norm =  np.array([contact_grad_norm_elastic, contact_grad_norm_target, contact_grad_norm_length, contact_grad_norm_smooth, contact_grad_norm_contact]).sum(axis=0)
    colors = [vs.elastic_color, vs.regularization_color, vs.target_color, vs.contact_color]
    labels = [vs.elastic_label, vs.regularization_label, vs.target_label, vs.contact_label]
    return contact_grad_norm_elastic, contact_grad_norm_target, contact_grad_norm_length, contact_grad_norm_smooth, contact_total_grad_norm, contact_grad_norm_contact, colors, labels

class Visualization_Setting():
    def __init__(self):
        self.cmap = plt.get_cmap("Set2")
        self.elastic_color = '#555358'
        self.target_color = self.cmap(1)
        self.rest_length_color = self.cmap(2)
        self.smoothness_color = self.cmap(3)
        self.regularization_color = self.cmap(2)
        self.curvature_color = self.cmap(4)
        self.contact_color = self.cmap(5)
        self.separation_color = self.cmap(5)
        self.tangential_color = self.cmap(6)
        self.joint_color = self.cmap(7)

        self.elastic_label = 'Elastic Energy'
        self.target_label = 'Target Surface Fitting'
        self.dist_surf_label = 'Avg. Dist. to Surface'
        self.rest_length_label = 'Rest Length Sum'
        self.smoothness_label = 'Curvature Variation'
        self.curvature_label = 'Curvature Sum'
        self.contact_label = 'Contact Forces'
        self.separation_label = 'Max. Separation Force'
        self.tangential_label = 'Max. Tangential Force'
        self.joint_label = 'Avg. Dist. to Nodes'
        self.regularization_label = 'Regularization'
        self.x_label = 'Iteration'
        self.figure_size = (17, 6)
        self.figure_label_size = 30

        self.stage_1_label = 'Stage 1'
        self.stage_2_label = 'Stage 2'
        self.stage_3_label = 'Stage 3'

def set_axis_limits(axis, data, ratio = 0.9):
    # There are nan inserted to indicate weight changes in stage 2, hence we use the min max that ignore the nans.
    low = np.nanmin(data)
    high = np.nanmax(data)
    span = high - low
    new_span = span / 0.9
    mid = (low + high) / 2.
    axis.set_ylim(mid - new_span / 2., mid + new_span / 2.)

def set_figure_label_and_limit(host, par1, par2, par3, par4, par5, par6, par7, p1, p2, p3, p4, p5, p6, p7, p8, combined_energy, combined_rest_length, combined_smoothing, combined_total_absolute_curvature, combined_distance_to_surface, combined_distance_to_joint, combined_separation_force, combined_tangential_force, vs):
    tkw = dict(size=4, width=1.5)

    if host != None:
        set_axis_limits(host, combined_energy)
        host.tick_params(axis='x', **tkw)
        host.get_yaxis().set_ticks([])
        host.get_yaxis().set_visible(False)
    if par1 != None:
        set_axis_limits(par1, combined_rest_length)
        par1.get_yaxis().set_ticks([])
        par1.get_yaxis().set_visible(False)
    if par2 != None:
        set_axis_limits(par2, combined_smoothing)
        par2.get_yaxis().set_ticks([])
        par2.get_yaxis().set_visible(False)
    if par3 != None:
        set_axis_limits(par3, combined_total_absolute_curvature)
        par3.get_yaxis().set_ticks([])
        par3.get_yaxis().set_visible(False)
    if par4 != None:
        set_axis_limits(par4, combined_distance_to_surface)
        par4.get_yaxis().set_ticks([])
        par4.get_yaxis().set_visible(False)
    if par5 != None:
        set_axis_limits(par5, combined_separation_force)
        par5.get_yaxis().set_ticks([])
        par5.get_yaxis().set_visible(False)
    if par6 != None:
        set_axis_limits(par6, combined_tangential_force)
        par6.get_yaxis().set_ticks([])
        par6.get_yaxis().set_visible(False)
    if par7 != None:
        set_axis_limits(par7, combined_distance_to_joint)
        par7.get_yaxis().set_ticks([])
        par7.get_yaxis().set_visible(False)

    # host.set_yscale('log')  

    # par5.set_yscale('log')
    # par6.set_yscale('log')

def plot_objective(vs, iterateData, total_objective, figure_name, label, iteration = None):
    fig, host = plt.subplots()
    cmap = plt.get_cmap("Set2")
    # colors = [vs.smoothness_color, vs.rest_length_color, vs.elastic_color]
    x=range(len(total_objective))
    # y=np.array([dps_objective_smooth, dps_objective_length, dps_objective_elastic])
    y=np.array(total_objective)
     
    # plt.stackplot(x,y, labels=[smoothness_label, rest_length_label, elastic_label], colors = colors)
    plt.plot(x,y)
    # plt.legend(loc='upper right', prop={'size': 15}, fancybox=True)
    #plt.show()
    fig.set_size_inches(vs.figure_size)
    plt.ylabel('Objective Value', fontsize = vs.figure_label_size)
    plt.title(label, fontsize = vs.figure_label_size)
    fig.set_size_inches(vs.figure_size)
    fig.savefig(figure_name, dpi=200)
    plt.close()

def plot_objective_stack(vs, iterateData, total_objective, objective_components_list, color_list, label_list, figure_name, label, weight_change_iters = [], grad_norm = False, iteration = None):
    if os.path.isfile(figure_name):
        return
    fig, host = plt.subplots()
    fig.set_size_inches(vs.figure_size)

    x=range(len(total_objective))
    y=np.array(objective_components_list)
     
    # Basic stacked area chart.
    plt.stackplot(x,y, labels=label_list, colors = color_list, baseline='zero')
    for iter_num in weight_change_iters:
        host.axvspan(iter_num - 1, iter_num, alpha=1, color='white')
    if iteration != None:
        host.axvline(iteration, alpha=1, color=vs.elastic_color)
    plt.legend(loc='upper right', prop={'size': 15}, fancybox=True)
    # host.set_xlabel(vs.x_label, fontsize = vs.figure_label_size)
    plt.ylabel('Grad Norm' if grad_norm else 'Objective Value', fontsize = vs.figure_label_size)
    plt.title(label, fontsize = vs.figure_label_size)
    # host.get_yaxis().set_ticks([])
    # host.get_yaxis().set_visible(False)
    fig.savefig(figure_name, bbox_inches='tight', dpi=200)
    plt.close()

def plot_ribbon_component_analysis(iterateData, elastic_energy, total_ribbon_length, smoothing, total_absolute_curvature, distance_to_surface, distance_to_joint, separation_force, tangential_force, DEBUG, combined_energy, combined_rest_length, combined_weighted_smoothing, combined_total_absolute_curvature, combined_distance_to_surface, combined_distance_to_joint, combined_separation_force, combined_tangential_force, vs, name, stage_index, label, iteration = -1):
    # Descriptive Stats
    figure_name_2D = '{}/video_images/{}_stage_{}_2D_{:03d}.{}'.format('{}/{}'.format(data_root, name), name, stage_index, iteration, 'svg' if use_svg else 'png')
    figure_name_3D = '{}/video_images/{}_stage_{}_3D_{:03d}.{}'.format('{}/{}'.format(data_root, name), name, stage_index, iteration, 'svg' if use_svg else 'png')
    if os.path.isfile(figure_name_2D):
        return
    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)
    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()
    par4 = host.twinx()
    par5 = host.twinx()
    par6 = host.twinx()
    par7 = host.twinx()

    n_iter = len(elastic_energy)
    p2, = par1.plot(range(n_iter), total_ribbon_length, linewidth = 3, color = vs.rest_length_color, label=vs.rest_length_label)
    p3, = par2.plot(range(n_iter), np.array(smoothing)[[0, int(len(iterateData)/4), int(3 * len(iterateData)/4), int(len(iterateData)-1)]], linewidth = 3, color = vs.smoothness_color, label=vs.smoothness_label) if DEBUG else par2.plot(range(n_iter), np.array(smoothing), linewidth = 3, color = vs.smoothness_color, label=vs.smoothness_label)
    p4, = par3.plot(range(n_iter), total_absolute_curvature, linewidth = 3, color = vs.curvature_color, label=vs.curvature_label)

    set_figure_label_and_limit(None, par1, par2, par3, None, None, None, None, None, p2, p3, p4, None, None, None, None, combined_energy, combined_rest_length, combined_weighted_smoothing, combined_total_absolute_curvature, combined_distance_to_surface, combined_distance_to_joint, combined_separation_force, combined_tangential_force, vs)

    host.set_xlabel(vs.x_label, fontsize = vs.figure_label_size)

    lines = [p2, p3, p4]
    if iteration != -1:
        host.axvline(iteration, alpha=1, color=vs.elastic_color)
    if stage_index == 3:
        # Legend
        leg = host.legend(lines, [l.get_label() for l in lines], loc="center right", facecolor='white', framealpha=1, fancybox=True, prop={'size': 15})
        bb = leg.get_bbox_to_anchor().inverse_transformed(host.transAxes)
        yOffset = -0.2
        bb.y0 += yOffset
        leg.set_bbox_to_anchor(bb, transform = host.transAxes)

    # plt.title(label, fontsize = vs.figure_label_size)
    fig.set_size_inches(vs.figure_size)

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='x', **tkw)
    host.get_yaxis().set_ticks([])
    host.get_yaxis().set_visible(False)
    par1.get_yaxis().set_ticks([])
    par1.get_yaxis().set_visible(False)
    par2.get_yaxis().set_ticks([])
    par2.get_yaxis().set_visible(False)
    par3.get_yaxis().set_ticks([])
    par3.get_yaxis().set_visible(False)
    par4.get_yaxis().set_ticks([])
    par4.get_yaxis().set_visible(False)
    par5.get_yaxis().set_ticks([])
    par5.get_yaxis().set_visible(False)
    par6.get_yaxis().set_ticks([])
    par6.get_yaxis().set_visible(False)
    par7.get_yaxis().set_ticks([])
    par7.get_yaxis().set_visible(False)

    fig.savefig(figure_name_2D, bbox_inches='tight', dpi=200)

    # Performance related stats
    fig, host = plt.subplots()
    fig.subplots_adjust(right=0.75)
    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()
    par4 = host.twinx()
    par5 = host.twinx()
    par6 = host.twinx()
    par7 = host.twinx()

    n_iter = len(elastic_energy)

    p1, = host.plot(range(n_iter), elastic_energy, linewidth = 3, color = vs.elastic_color, label=vs.elastic_label)
    p8, = par7.plot(range(n_iter), distance_to_joint, linewidth = 3, color = vs.joint_color, label=vs.joint_label)

    p5, = par4.plot(range(n_iter), distance_to_surface, linewidth = 3, color = vs.target_color, label=vs.dist_surf_label)

    # if stage_index == 1:
    #     p6, p7, par5, par6 = None, None, None, None
    # else:
    p6, = par5.plot(range(n_iter), separation_force, linewidth = 3, color = vs.separation_color, label=vs.separation_label)
    p7, = par6.plot(range(n_iter), tangential_force, linewidth = 3, color = vs.tangential_color, label=vs.tangential_label)

    set_figure_label_and_limit(host, None, None, None, par4, par5, par6, par7, p1, None, None, None, p5, p6, p7, p8, combined_energy, combined_rest_length, combined_weighted_smoothing, combined_total_absolute_curvature, combined_distance_to_surface, combined_distance_to_joint, combined_separation_force, combined_tangential_force, vs)
    # host.set_xlabel(vs.x_label, fontsize = vs.figure_label_size)

    lines = [p1, p5, p6, p7, p8]
    if iteration != -1:
        host.axvline(iteration, alpha=1, color=vs.elastic_color)
    if stage_index == 3:
        # Legend
        leg = host.legend(lines, [l.get_label() for l in lines], loc="center right", facecolor='white', framealpha=1, fancybox=True, prop={'size': 15})
        bb = leg.get_bbox_to_anchor().inverse_transformed(host.transAxes)
        yOffset = -0.4
        bb.y0 += yOffset
        leg.set_bbox_to_anchor(bb, transform = host.transAxes)
    fig.set_size_inches(vs.figure_size)

    plt.title(label, fontsize = vs.figure_label_size)

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='x', **tkw)
    host.get_yaxis().set_ticks([])
    host.get_yaxis().set_visible(False)
    par1.get_yaxis().set_ticks([])
    par1.get_yaxis().set_visible(False)
    par2.get_yaxis().set_ticks([])
    par2.get_yaxis().set_visible(False)
    par3.get_yaxis().set_ticks([])
    par3.get_yaxis().set_visible(False)
    par4.get_yaxis().set_ticks([])
    par4.get_yaxis().set_visible(False)
    par5.get_yaxis().set_ticks([])
    par5.get_yaxis().set_visible(False)
    par6.get_yaxis().set_ticks([])
    par6.get_yaxis().set_visible(False)
    par7.get_yaxis().set_ticks([])
    par7.get_yaxis().set_visible(False)
    fig.savefig(figure_name_3D, bbox_inches='tight', dpi=200)
    plt.close()

def plot_merged_ribbon_component_analysis(iterateData, elastic_energy, total_ribbon_length, smoothing, total_absolute_curvature, distance_to_surface, distance_to_joint, separation_force, tangential_force, DEBUG, combined_energy, combined_rest_length, combined_weighted_smoothing, combined_total_absolute_curvature, combined_distance_to_surface, combined_distance_to_joint, combined_separation_force, combined_tangential_force, vs, name, stage_index, label, iteration = None):
    figure_name = '{}/video_images/{}_stage_{}_components_description_{:03d}.{}'.format('{}/{}'.format(data_root, name), name, stage_index, iteration, 'svg' if use_svg else 'png')
    if os.path.isfile(figure_name):
        return
    # Descriptive Stats
    fig, host = plt.subplots()
    fig.set_size_inches(vs.figure_size)
    par1 = host.twinx()
    par2 = host.twinx()
    par3 = host.twinx()
    par4 = host.twinx()
    par5 = host.twinx()
    par6 = host.twinx()
    par7 = host.twinx()

    n_iter = len(elastic_energy)
    p2, = par1.plot(range(n_iter), total_ribbon_length, linewidth = 3, color = vs.rest_length_color, label=vs.rest_length_label)
    p4, = par3.plot(range(n_iter), total_absolute_curvature, linewidth = 3, color = vs.curvature_color, label=vs.curvature_label)
    p8, = par7.plot(range(n_iter), distance_to_joint, linewidth = 3, color = vs.joint_color, label=vs.joint_label)

    p1, = host.plot(range(n_iter), elastic_energy, linewidth = 3, color = vs.elastic_color, label=vs.elastic_label)
    p3, = par2.plot(range(n_iter), np.array(smoothing)[[0, int(len(iterateData)/4), int(3 * len(iterateData)/4), int(len(iterateData)-1)]], linewidth = 3, color = vs.smoothness_color, label=vs.smoothness_label) if DEBUG else par2.plot(range(n_iter), np.array(smoothing), linewidth = 3, color = vs.smoothness_color, label=vs.smoothness_label)
    p5, = par4.plot(range(n_iter), distance_to_surface, linewidth = 3, color = vs.target_color, label=vs.dist_surf_label)
    p6, = par5.plot(range(n_iter), separation_force, linewidth = 3, color = vs.separation_color, label=vs.separation_label)
    p7, = par6.plot(range(n_iter), tangential_force, linewidth = 3, color = vs.tangential_color, label=vs.tangential_label)
    set_figure_label_and_limit(host, par1, par2, par3, par4, par5, par6, par7, p1, p2, p3, p4, p5, p6, p7, p8, combined_energy, combined_rest_length, combined_weighted_smoothing, combined_total_absolute_curvature, combined_distance_to_surface, combined_distance_to_joint, combined_separation_force, combined_tangential_force, vs)

    host.set_xlabel(vs.x_label, fontsize = vs.figure_label_size)

    lines = [p1, p2, p3, p4, p5, p6, p7, p8]
    if iteration != None:
        host.axvline(iteration, alpha=1, color=vs.elastic_color)
    # Legend
    if stage_index == 1:
        leg = host.legend(lines, [l.get_label() for l in lines], loc="center left", facecolor='white', framealpha=1, fancybox=True, prop={'size': 15})
        bb = leg.get_bbox_to_anchor().inverse_transformed(host.transAxes)
        yOffset = -0.2
        bb.y0 += yOffset
        leg.set_bbox_to_anchor(bb, transform = host.transAxes)

    # plt.title(label, fontsize = vs.figure_label_size)
    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='x', **tkw)
    host.get_yaxis().set_ticks([])
    host.get_yaxis().set_visible(False)
    fig.savefig(figure_name, bbox_inches='tight', dpi=200)

def freedRibbonEquilibriumVisualization(l, ribbonIdx, vmin=None, vmax=None):
    """
    Visualizes what happens when a single ribbon of a woven linkage `l` is freed from the crossing constraints.
    Maximum bending stresses are visualized in the constrained and unconstrained configurations.
    """
    ribbons = order_segments_by_ribbons(l)
    rod, fixedVars = construct_elastic_rod_loop_from_rod_segments(l, ribbons[ribbonIdx])
    rod.setMaterial(l.segment(0).rod.material(0))

    rod_eq = copy.deepcopy(rod)

    # Clamp first and last edge
    fv = [0, 1, 2, 3, 4, 5]
    fv += list(3 * (rod_eq.numVertices() - 2) + np.arange(6))
    fv += [rod_eq.thetaOffset(), rod_eq.thetaOffset() + rod_eq.numEdges() - 1]

    from io_redirection import suppress_stdout
    with suppress_stdout(): elastic_rods.compute_equilibrium(rod_eq, fixedVars=fv)

    import tri_mesh_viewer, mesh_operations, mesh

    def rodStressVis(rod): return rod.visualizationField(rod.maxBendingStresses())
    def watertightMesh(rod): return mesh_operations.mergedMesh([mesh.Mesh(*rod.visualizationGeometry()[0:2])], [rodStressVis(rod)])

    mergedVisData = [watertightMesh(r) for r in [rod, rod_eq]]
    m = mesh.Mesh(*mesh_operations.concatenateMeshes([d[0:2] for d in mergedVisData]))
    stresses = np.hstack([np.array(d[2]).ravel() for d in mergedVisData])

    tsf = l.get_target_surface_fitter()
    view = linkage_vis.LinkageViewerWithSurface(m, mesh.Mesh(tsf.V, tsf.F), width=768, height=512)
    view.viewOptions[linkage_vis.LinkageViewerWithSurface.ViewType.SURFACE].transparent = True
    view.applyViewOptions()
    sf = vis.fields.ScalarField(m, stresses, colormap=cm.magma, vmin=vmin, vmax=vmax)
    view.showScalarField(sf)
    return view

def render_video(model_name, stage_label, ribbon_width, renderCam, framerate = 5):
    '''
    Render the equilibrium ribbon results offscreen for each iteration and combine them into a video. 
    '''
    from PIL import Image
    import ffmpeg
    model_path_base = 'optimization_diagram_results/{}'.format(model_name)
    video_folder = os.path.join(model_path_base, 'video')
    image_folder = os.path.join(video_folder, '{}_images'.format(stage_label))
    if not os.path.exists(image_folder):
        os.makedirs(image_folder) 
    np.save(os.path.join(video_folder, 'renderCam.npy'), renderCam)

    def renderToFile(view, renderCam, path):
        orender = view.offscreenRenderer(width=2048, height=2048)
        orender.setCameraParams(renderCam)
        orender.render()
        orender.save(path)
    def write_distance_image(i):
        image_name = os.path.join(image_folder, 'iter_{:03d}.png'.format(i))
        if not os.path.isfile(image_name):
            curved_linkage = pickle.load(gzip.open(os.path.join(model_path_base, 'pickle/{}_{}.pkl.gz'.format(stage_label.split('_')[1], i)), 'r'))
            curved_linkage_view = linkage_vis.LinkageViewer(curved_linkage)
            distance_color = write_distance_to_linkage_mesh(curved_linkage, ribbon_width, None, return_distance_field = True)
            curved_linkage_view.update(scalarField = distance_color[:, :3])
            renderToFile(curved_linkage_view, renderCam, image_name)
    def convert_png_to_jpg(i):
        image_name = os.path.join(image_folder, 'iter_{:03d}.png'.format(i))
        new_image_name = os.path.join(video_folder, '{}_images/iter_{:03d}.jpg'.format(stage_label, i))
        if not os.path.isfile(new_image_name):
            im = Image.open(image_name)
            bg = Image.new('RGB', im.size, (255,255,255))
            bg.paste(im,im)
            bg.save(new_image_name)
    iterateData = np.load(os.path.join(model_path_base, '{}_{}.npy'.format(model_name, stage_label)), allow_pickle=True)
    [write_distance_image(i) for i in range(len(iterateData))]
    [convert_png_to_jpg(i) for i in range(len(iterateData))]
    video_name = os.path.join(video_folder, '{}.mp4'.format(stage_label))
    if os.path.isfile(video_name):
        os.remove(video_name)
    (ffmpeg
        .input(os.path.join(image_folder, 'iter_*.jpg'), pattern_type='glob', framerate=framerate)
        .output(video_name, crf=20, preset='slower', movflags='faststart', pix_fmt='yuv420p')
        .run()
    )

def combine_three_stage_video(model_name):
    '''
    Combine the video for each stage into one video.
    '''
    import ffmpeg
    model_path_base = 'optimization_diagram_results/{}'.format(model_name)
    video_folder = os.path.join(model_path_base, 'video')
    file_list_path = os.path.join(video_folder, 'video_file_list.txt')
    in1 = '{}.mp4'.format('stage_1')
    in2 = '{}.mp4'.format('stage_2')
    in3 = '{}.mp4'.format('stage_3')
    with open(file_list_path, 'w') as f:
        f.write('file {}\nfile {}\nfile {}'.format(in1, in2, in3))
    video_name = os.path.join(video_folder, 'combined.mp4')
    if os.path.isfile(video_name):
        os.remove(video_name)
    (ffmpeg
        .input(file_list_path, format='concat', safe=0)
        .output(video_name)
        .run()
    )