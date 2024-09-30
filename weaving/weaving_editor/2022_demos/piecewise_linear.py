#!/usr/bin/env python
# coding: utf-8

elastic_rods_dir = '../../../elastic_rods/python/'
weaving_dir = '../../'
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

from pipeline_helper import (initialize_linkage, get_normal_deviation, set_joint_vector_field, stage_1_optimization, initialize_stage_2_optimizer, stage_2_optimization, InputOrganizer, write_all_output, set_surface_view_options, get_structure_analysis_view, get_max_distance_to_target_surface, contact_optimization, show_selected_joints, highlight_rod_and_joint)

import vis.fields
import matplotlib.cm as cm
import time

import parallelism
parallelism.set_max_num_tbb_threads(12)
parallelism.set_hessian_assembly_num_threads(4)
parallelism.set_gradient_assembly_num_threads(4)


import json

with open(osp.join(weaving_dir + 'woven_model.json')) as f:
    data = json.load(f)

model_info = data['models'][0]

thickness, width, name, use_constant_width, width_scale, scale_joint_weight, update_attraction_weight, number_of_updates, fix_boundary, only_two_stage = model_info['thickness'], model_info['width'], model_info['name'], model_info['constant_cross_section'], model_info['cross_section_scale'], model_info['scale_joint_weight'], model_info['update_attraction_weight'], model_info['number_of_updates'], model_info['fix_boundary'], model_info['only_two_stage']
joint_weight, scale, joint_list = 0, 0, []
if float(scale_joint_weight.split(', ')[0]) != -1:
    joint_weight, scale, joint_list = float(scale_joint_weight.split(', ')[0]), float(scale_joint_weight.split(', ')[1]), [int(x) for x in scale_joint_weight.split(', ')[2:]]
    
io = InputOrganizer(name, thickness, width, weaving_dir)

fixed_boundary_joints = []

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
drw = 0.001
dsw = 0.01


curved_linkage = initialize_linkage(surface_path = io.SURFACE_PATH, useCenterline = True, model_path = io.MODEL_PATH, cross_section = io.RIBBON_CS, subdivision_res = io.SUBDIVISION_RESOLUTION, interleaving_type=InterleavingType.weaving, use_constant_width = True)
curved_linkage.set_design_parameter_config(use_restLen = True, use_restKappa = True)
curved_save_tgt_joint_pos = curved_linkage.jointPositions()
curved_linkage.attraction_weight = 1e-5
curved_linkage_view = linkage_vis.LinkageViewerWithSurface(curved_linkage, io.SURFACE_PATH)
set_surface_view_options(curved_linkage_view)
curved_linkage_view.show()


# ### Stage 1

iterateData, _ = stage_1_optimization(curved_linkage, drw, dsw, curved_linkage_view)

def eqm_callback(prob, i):
    curved_linkage_view.update()

with so(): elastic_rods.compute_equilibrium(curved_linkage, callback = eqm_callback, options = OPTS)