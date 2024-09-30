elastic_rods_dir = '../../elastic_rods/python/'
weaving_dir = '../'
import os
import os.path as osp
import sys; sys.path.append(elastic_rods_dir); sys.path.append(weaving_dir)
import numpy as np, elastic_rods
import numpy.linalg as la
from bending_validation import suppress_stdout as so
import matplotlib.pyplot as plt
import MeshFEM, py_newton_optimizer
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


import vis.fields
import matplotlib.cm as cm
import time

import cross_section_scaling
importlib.reload(cross_section_scaling)

try:
    import linkage_optimization
    from linkage_optimization import OptEnergyType
except ImportError:
    pass

import structural_analysis
import pickle, gzip
import json

rod_length = 0.3534025445286393
width = rod_length / 15 * 5
thickness = width / 5 * 0.35

# Sphere 1
default_camera_parameters = ((3.466009282140468, -4.674139805388271, -2.556131049738206), (-0.21402574298422497, -0.06407538766530313, -0.9747681088523519),(0.1111, 0.1865, 0.5316))
RIBBON_CS = [thickness, width]
MODEL_NAME = "sphere_1"
MODEL_PATH = osp.join(weaving_dir + 'normalized_objs/models/{}.obj'.format(MODEL_NAME))
SUBDIVISION_RESOLUTION = 20
INPUT_SURFACE_PATH = osp.join(weaving_dir + 'normalized_objs/surface_models/{}.obj'.format(MODEL_NAME))
RIBBON_NAME = "{}_strip".format(MODEL_NAME)

prev_vars = None
prev_time_stamp = None

# Optimization parameters.
OPTS = py_newton_optimizer.NewtonOptimizerOptions()
OPTS.gradTol = 1e-8
OPTS.verbose = 1;
OPTS.beta = 1e-8
OPTS.niter = 200
OPTS.verboseNonPosDef = False

class InputOrganizer():
    def __init__(self, model_name, thickness, width, weaving_dir):
        self.default_camera_parameters = ((3.466009282140468, -4.674139805388271, -2.556131049738206), (-0.21402574298422497, -0.06407538766530313, -0.9747681088523519),(0.1111, 0.1865, 0.5316))
        self.RIBBON_CS = [thickness, width]
        self.MODEL_NAME = model_name
        self.MODEL_PATH = osp.join(weaving_dir + 'normalized_objs/models/{}.obj'.format(model_name))
        self.SUBDIVISION_RESOLUTION = 20
        self.SURFACE_PATH = osp.join(weaving_dir + 'normalized_objs/surface_models/{}.obj'.format(model_name))
        self.RIBBON_NAME = "{}_ribbons".format(model_name)

def initialize_linkage(surface_path = INPUT_SURFACE_PATH, useCenterline = True, cross_section = RIBBON_CS, subdivision_res = SUBDIVISION_RESOLUTION, model_path = MODEL_PATH, use_constant_width = True, width_scale = [0.5, 2], interleaving_type = InterleavingType.triaxialWeave, input_joint_normals = []):
    if surface_path == None:
        l = elastic_rods.RodLinkage(model_path, subdivision_res, False, interleaving_type, [], input_joint_normals)
    else:
        l = elastic_rods.SurfaceAttractedLinkage(surface_path, useCenterline, model_path, subdivision_res, False, interleaving_type, [], input_joint_normals)
    if use_constant_width:
        l.setMaterial(elastic_rods.RodMaterial('rectangle', 2000, 0.3, cross_section, stiffAxis=elastic_rods.StiffAxis.D1))
    else:
        cross_section_scaling.apply_density_based_cross_sections(l, elastic_rods.CrossSection.construct('rectangle', 2000, 0.3, [cross_section[0], width_scale[0] * cross_section[1]]), elastic_rods.CrossSection.construct('rectangle', 2000, 0.3, [cross_section[0], width_scale[1] * cross_section[1]]))
    if surface_path != None:
        l.set_holdClosestPointsFixed(True);
        l.set_attraction_tgt_joint_weight(0.01);
        l.attraction_weight = 100;
    return l

def stage_1_optimization(linkage, rest_length_weight, smoothing_weight, linkage_view, update_color = False, niter = 10000, E0 = -1, l0 = -1):
    global prev_vars
    global prev_time_stamp
    iterateData = []
    def callback(prob, i):
        global prev_vars
        global prev_time_stamp
        curr_vars = linkage.getExtendedDoFsPSRL()
        iterateData.append({'iteration_time': time.time() - prev_time_stamp,
                            'weighted_energy' : prob.weighted_energy(),
                            'weighted_smoothness' : prob.weighted_smoothness(),
                            'weighted_length' : prob.weighted_length(),
                            'extendedDoFsPSRL' : curr_vars})
        prev_time_stamp = time.time()
        prev_vars[:] = curr_vars

        if linkage_view:
            if (i % 5) != 0: return
            if update_color:
                bottomColor =[79/255., 158/255., 246/255.]
                topColor =[0.5, 0.5, 0.5]
                heights = linkage.visualizationGeometryHeightColors()
                colors = np.take(np.array([bottomColor, topColor]), heights < heights.mean(), axis=0)
                linkage_view.update(scalarField=colors)
            else:
                linkage_view.update()

    dpo = elastic_rods.get_designParameter_optimizer(linkage, rest_length_weight, smoothing_weight, callback = callback, E0 = E0, l0 = l0)
    dpo.options.niter = niter

    prev_time_stamp = time.time()
    prev_vars = linkage.getExtendedDoFsPSRL()
    # # Gather starting point information.
    # callback(elastic_rods.designParameter_problem(linkage, rest_length_weight, smoothing_weight, E0 = E0, l0 = l0), 0)
    with so(): cr = dpo.optimize()
    return iterateData, dpo

def get_fixed_boundary_joint(linkage):
    fixed_boundary_joint = []
    joint_pos = linkage.designParameterSolveFixedVars()
    for ji in range(linkage.numJoints()):
        if linkage.joint(ji).valence() < 4:
            fixed_boundary_joint.extend(joint_pos[ji * 3: ji * 3 + 3])
    return fixed_boundary_joint

def initialize_stage_2_optimizer(linkage, input_surface_path, tgt_joint_pos, linkage_view, rest_length_weight, smoothing_weight, fixed_boundary_joint = [], equilibrium_iter = 25, inner_gradTol = 1e-6):
    OPTS = py_newton_optimizer.NewtonOptimizerOptions()
    OPTS.gradTol = inner_gradTol
    OPTS.verbose = 1
    OPTS.beta = 1e-8
    OPTS.niter = equilibrium_iter
    OPTS.verboseNonPosDef = False

    useCenterline = True
    optimizer = linkage_optimization.WeavingOptimization(linkage, input_surface_path, useCenterline, OPTS, pinJoint = 0, useFixedJoint = False, fixedVars = fixed_boundary_joint)
    optimizer.set_target_joint_position(tgt_joint_pos)
    if linkage_view:
        linkage_view.update()

    optimizer.rl_regularization_weight = rest_length_weight
    optimizer.smoothing_weight = smoothing_weight
    optimizer.beta = 500000.0
    optimizer.gamma = 1
    return optimizer

# Used for grasshopper pipeline
def initialize_stage_2_optimizer_gh(linkage, target_vertices, target_faces, tgt_joint_pos, linkage_view, rest_length_weight, smoothing_weight, fixed_boundary_joint = [], equilibrium_iter = 25, inner_gradTol = 1e-6):
    OPTS = py_newton_optimizer.NewtonOptimizerOptions()
    OPTS.gradTol = inner_gradTol
    OPTS.verbose = 1
    OPTS.beta = 1e-8
    OPTS.niter = equilibrium_iter
    OPTS.verboseNonPosDef = False

    useCenterline = True
    optimizer = linkage_optimization.WeavingOptimization(linkage, target_vertices, target_faces, useCenterline, OPTS, pinJoint = 0, useFixedJoint = False, fixedVars = fixed_boundary_joint)
    optimizer.set_target_joint_position(tgt_joint_pos)
    if linkage_view:
        linkage_view.update()

    optimizer.rl_regularization_weight = rest_length_weight
    optimizer.smoothing_weight = smoothing_weight
    optimizer.beta = 500000.0
    optimizer.gamma = 1
    return optimizer

def get_component_gradient_norm(optimizer, curr_type):
    save_beta = optimizer.beta
    save_gamma = optimizer.gamma
    save_smoothing_weight = optimizer.smoothing_weight
    save_rl_regularization_weight = optimizer.rl_regularization_weight
    save_contact_force_weight = optimizer.contact_force_weight

    optimizer.beta = 0
    optimizer.gamma = 0
    optimizer.smoothing_weight = 0
    optimizer.rl_regularization_weight = 0
    optimizer.contact_force_weight = 0
    if curr_type == OptEnergyType.Full:
        optimizer.beta = save_beta
        optimizer.gamma = save_gamma
        optimizer.smoothing_weight = save_smoothing_weight
        optimizer.rl_regularization_weight = save_rl_regularization_weight
        optimizer.contact_force_weight = save_contact_force_weight
    elif curr_type == OptEnergyType.Target:
        optimizer.beta = save_beta
    elif curr_type == OptEnergyType.Smoothing:
        optimizer.smoothing_weight = save_smoothing_weight
    elif curr_type == OptEnergyType.Regularization:
        optimizer.rl_regularization_weight = save_rl_regularization_weight
    elif curr_type == OptEnergyType.ElasticBase:
        optimizer.gamma = save_gamma
    elif curr_type == OptEnergyType.ContactForce:
        optimizer.contact_force_weight = save_contact_force_weight

    optimizer.invalidateAdjointState()
    grad_norm = la.norm(optimizer.gradp_J(optimizer.params()))

    optimizer.beta = save_beta
    optimizer.gamma = save_gamma
    optimizer.smoothing_weight = save_smoothing_weight
    optimizer.rl_regularization_weight = save_rl_regularization_weight
    optimizer.contact_force_weight = save_contact_force_weight
    optimizer.invalidateAdjointState()
    return grad_norm

class WeavingOptimizationCallback:
    def __init__(self, optimizer, linkage, linkage_view, update_color = False, no_surface = False, callback_freq = 1):
        self.optimizer     = optimizer
        self.linkage       = linkage
        self.linkage_view  = linkage_view
        self.update_color = update_color
        self.no_surface    = no_surface
        self.callback_freq = callback_freq
        self.iterateData = []

    def __call__(self):
        global prev_vars
        global prev_time_stamp
        #print('running callback', flush=True)

        if (self.no_surface):
            self.linkage_view.update()
            return
        #print('here1', flush=True)

        # Why don't we also do this in the "no_surface" callback?
        curr_vars = self.linkage.getExtendedDoFsPSRL()
        # Record values of all objective terms, plus timestamp and variables.
        idata = {t.name: t.term.value() for t in self.optimizer.objective.terms}
        idata.update({'iteration_time':   time.time() - prev_time_stamp,
                      'extendedDoFsPSRL': curr_vars}) 
        idata.update({'{}_grad_norm'.format(t.name): get_component_gradient_norm(self.optimizer, t.type) for t in self.optimizer.objective.terms})
        self.iterateData.append(idata)
        prev_time_stamp = time.time()
        
        if self.linkage_view and (len(self.iterateData) % self.callback_freq == 0):
            if self.update_color:
                bottomColor =[79/255., 158/255., 246/255.]
                topColor =[0.5, 0.5, 0.5]
                heights = self.linkage.visualizationGeometryHeightColors()
                colors = np.take(np.array([bottomColor, topColor]), heights < heights.mean(), axis=0)
                self.linkage_view.update(scalarField=colors)
            else:
                pass
                self.linkage_view.update()

    def numIterations(self): return len(self.iterateData)


def level_target_weight(optimizer, linkage, fixed_boundary_joints):
    with so(): elastic_rods.compute_equilibrium(linkage, callback = None, options = OPTS, fixedVars = fixed_boundary_joints)
    if optimizer.objective.terms[0].term.value() > optimizer.objective.terms[1].term.value() and get_max_distance_to_target_surface(linkage) > 0.1:
            print('elastic energy', optimizer.objective.terms[0].term.value(), 'Target', optimizer.objective.terms[1].term.value())
            optimizer.objective.terms[1].term.weight *= optimizer.objective.terms[0].term.value() / optimizer.objective.terms[1].term.value()
    if optimizer.objective.terms[2].term.value() + optimizer.objective.terms[3].term.value() > optimizer.objective.terms[1].term.value():
        optimizer.objective.terms[1].term.weight *= (optimizer.objective.terms[2].term.value() + optimizer.objective.terms[3].term.value()) / optimizer.objective.terms[1].term.value()
    return optimizer.objective.terms[1].term.weight

def level_regularization_weight_with_target(optimizer):
    print('level_regularization', optimizer.objective.terms[2].term.value() + optimizer.objective.terms[3].term.value(), optimizer.objective.terms[1].term.value())
    if optimizer.objective.terms[2].term.value() + optimizer.objective.terms[3].term.value() > optimizer.objective.terms[1].term.value():
        optimizer.objective.terms[2].term.weight *= optimizer.objective.terms[1].term.value() / (optimizer.objective.terms[2].term.value() + optimizer.objective.terms[3].term.value())
        optimizer.objective.terms[3].term.weight *= optimizer.objective.terms[1].term.value() / (optimizer.objective.terms[2].term.value() + optimizer.objective.terms[3].term.value())
    return [optimizer.objective.terms[2].term.weight, optimizer.objective.terms[3].term.weight]

def stage_2_optimization(optimizer, linkage, input_surface_path, tgt_joint_pos, linkage_view, minRestLen, fixed_boundary_joints, update_attraction_weight = -5, num_updates = 2, return_weight_change_iteration = False, update_color = False, no_surface = False, callback_freq = 1, maxIter = 2000):
    global prev_vars
    global prev_time_stamp

    algorithm = linkage_optimization.OptAlgorithm.NEWTON_CG
    prev_vars = linkage.getExtendedDoFsPSRL()

    print('rl_regularization_weight', optimizer.rl_regularization_weight)
    print('smoothing_weight', optimizer.smoothing_weight)
    optimizer.set_holdClosestPointsFixed(False)
    weight_change_iterations = []

    cb = WeavingOptimizationCallback(optimizer, linkage, linkage_view, update_color, no_surface, callback_freq)

    if no_surface:
        prev_time_stamp = time.time()
        optimizer.setLinkageAttractionWeight(0) # Why isn't this done in contact optimization?
        solverStatus = optimizer.WeavingOptimize(algorithm, maxIter, 1.0, 1e-2, cb, minRestLen)
        return optimizer, cb.iterateData, solverStatus

    prev_time_stamp = time.time()
    # Run stage 2 optimization with weight scheduling.
    target_weight = []
    solverStatusList = []
    if num_updates == 1:
        curr_update_weight = np.logspace(update_attraction_weight, update_attraction_weight, num_updates)
        weight_change_iterations.append(cb.numIterations())
        print('stage 2 optimization with attraction weight {}'.format(curr_update_weight[0]), flush=True)
        prev_time_stamp = time.time()
        optimizer.setLinkageAttractionWeight(curr_update_weight[0])
        target_weight.append([level_target_weight(optimizer, linkage, fixed_boundary_joints)])
        # Gather the starting point information.
        cb()
        solverStatus = optimizer.WeavingOptimize(algorithm, maxIter, 1.0, 1e-2, cb, minRestLen)
        solverStatusList.append(solverStatus)
    else:
        for curr_update_weight in np.logspace(2, update_attraction_weight, num_updates):
            weight_change_iterations.append(cb.numIterations())
            print('stage 2 optimization with attraction weight {}'.format(curr_update_weight), flush=True)
            prev_time_stamp = time.time()
            optimizer.setLinkageAttractionWeight(curr_update_weight)
            target_weight.append([level_target_weight(optimizer, linkage, fixed_boundary_joints)])
            # Gather the starting point information.
            cb()
            solverStatus = optimizer.WeavingOptimize(algorithm, maxIter, 1.0, 1e-2, cb, minRestLen)
            solverStatusList.append(solverStatus)
    if return_weight_change_iteration:
        return optimizer, cb.iterateData, weight_change_iterations, target_weight, solverStatusList
    return optimizer, cb.iterateData, solverStatusList

def level_contact_weight(optimizer):
    if optimizer.objective.terms[0].term.value() > optimizer.objective.terms[-1].term.value() or optimizer.objective.terms[1].term.value() > optimizer.objective.terms[-1].term.value():
        optimizer.objective.terms[0].term.weight *=  optimizer.objective.terms[-1].term.value() / max(optimizer.objective.terms[0].term.value(), optimizer.objective.terms[1].term.value())
        optimizer.objective.terms[1].term.weight *=  optimizer.objective.terms[-1].term.value() / max(optimizer.objective.terms[0].term.value(), optimizer.objective.terms[1].term.value())
    if optimizer.objective.terms[2].term.value() + optimizer.objective.terms[3].term.value() > optimizer.objective.terms[-1].term.value():
        optimizer.objective.terms[-1].term.weight *= (optimizer.objective.terms[2].term.value() + optimizer.objective.terms[3].term.value()) / optimizer.objective.terms[-1].term.value()

def level_regularization_weight_with_contact(optimizer):
    if optimizer.objective.terms[2].term.value() + optimizer.objective.terms[3].term.value() > optimizer.objective.terms[-1].term.value():
        optimizer.objective.terms[2].term.weight *= optimizer.objective.terms[-1].term.value() / (optimizer.objective.terms[2].term.value() + optimizer.objective.terms[3].term.value())
        optimizer.objective.terms[3].term.weight *= optimizer.objective.terms[-1].term.value() / (optimizer.objective.terms[2].term.value() + optimizer.objective.terms[3].term.value())

def contact_optimization(optimizer, linkage, linkage_view, minRestLen, contact_weight = 1e4, normalActivationThreshold = -1e-4, update_color = False, no_surface = False, normalWeight = 1, tangentialWeight = 1, torqueWeight = 1, useBoundary = False, maxIter = 2000, update_attraction_weight = -5, callback_freq = 1):
    global prev_vars
    global prev_time_stamp

    algorithm = linkage_optimization.OptAlgorithm.NEWTON_CG
    prev_vars = linkage.getExtendedDoFsPSRL()

    optimizer.set_holdClosestPointsFixed(False)
    if no_surface:
        optimizer.setLinkageAttractionWeight(0)
    else:
        curr_update_weight = np.logspace(update_attraction_weight, update_attraction_weight, 1)
        optimizer.setLinkageAttractionWeight(curr_update_weight)

    cb = WeavingOptimizationCallback(optimizer, linkage, linkage_view, update_color, no_surface, callback_freq = callback_freq)

    algorithm = linkage_optimization.OptAlgorithm.NEWTON_CG
    optimizer.objective.terms[-1].term.weight = contact_weight
    optimizer.objective.terms[-1].term.normalWeight = normalWeight
    optimizer.objective.terms[-1].term.tangentialWeight = tangentialWeight
    optimizer.objective.terms[-1].term.torqueWeight = torqueWeight
    optimizer.objective.terms[-1].term.normalActivationThreshold = normalActivationThreshold
    if not useBoundary:
        optimizer.objective.terms[-1].term.boundaryNormalWeight = 0
        optimizer.objective.terms[-1].term.boundaryTangentialWeight = 0
        optimizer.objective.terms[-1].term.boundaryTorqueWeight = 0

    level_contact_weight(optimizer)
    prev_time_stamp = time.time()
    # Gather starting point information.
    cb()
    prev_vars = linkage.getExtendedDoFsPSRL()
    # Run contact optimization.
    solverStatus = optimizer.WeavingOptimize(algorithm, maxIter, 1.0, 1e-2, cb, minRestLen)
    return optimizer, cb.iterateData, solverStatus

def write_all_output(linkage, resolution, model_name, ribbon_name, use_family_label, scale, thickness, target_width = 5, flip_angles = False, width = 5, use_surface = True, write_stress = False):
    if not os.path.exists('results/{}'.format(model_name)):
        os.makedirs('results/{}'.format(model_name))    
    os.chdir('./results/{}'.format(model_name))
    ribbons = order_segments_by_ribbons(linkage)
    write_linkage_ribbon_output_florin(linkage, ribbons, resolution, model_name, use_family_label = use_family_label, scale = scale, write_stress = write_stress)
    get_all_curve_pattern(linkage, thickness, resolution, ribbon_name, 'svg', target_ribbon_width = target_width, flip_angles = flip_angles)
    np.save('{}_dof.npy'.format(model_name), linkage.getExtendedDoFsPSRL())
    if use_surface:
        write_distance_to_linkage_mesh(linkage, width, model_name)
        # write_centerline_normal_deviation_to_linkage_mesh(linkage, model_name)

# Analysis
def get_normal_deviation(linkage):
    joint_normals = np.array([linkage.joint(i).normal for i in range(linkage.numJoints())])
    joint_projection_normals = linkage.get_closest_point_normal(linkage.jointPositions())
    joint_projection_normals = joint_projection_normals.reshape(joint_normals.shape)
    deviation = []
    deviation_vector = []
    for i in range(len(joint_normals)):
        closeness = np.dot(joint_projection_normals[i], joint_normals[i])
        deviation.append(np.arccos(abs(closeness)) / np.pi * 180)
        if closeness < 0:
            deviation_vector.append(joint_projection_normals[i] + joint_normals[i])
        else:
            deviation_vector.append(joint_projection_normals[i] - joint_normals[i])
    return deviation, deviation_vector, joint_normals, joint_projection_normals

def set_joint_vector_field(linkage, linkage_view, joint_vector_field):
    vector_field = [np.zeros((s.rod.numVertices(), 3)) for s in linkage.segments()]
    for ji in range(linkage.numJoints()):
        seg_index = linkage.joint(ji).segments_A[0]
        if linkage.segment(seg_index).startJoint == ji:
            vx_index = 0
        else:
            vx_index = -1
        vector_field[seg_index][vx_index] = joint_vector_field[ji]
    
    linkage_view.update(vectorField = vector_field)

def show_selected_joints(linkage, joint_list, flip = False):
    joint_vector_field = [np.zeros((s.rod.numVertices(), 3)) for s in linkage.segments()]
    for ji in range(linkage.numJoints()):
        if ji in joint_list:
            seg_index = linkage.joint(ji).segments_A[0]
            if linkage.segment(seg_index).startJoint == ji:
                vx_index = 0
            else:
                vx_index = -1
            joint_vector_field[seg_index][vx_index] = linkage.segment(seg_index).rod.deformedConfiguration().materialFrame[vx_index].d2
            if flip:
                joint_vector_field[seg_index][vx_index] *= -1
    return joint_vector_field

def get_structure_analysis_view(linkage):
    structural_analysis.weavingCrossingAnalysis(linkage, omitBoundary=True)
    v = structural_analysis.crossingForceFieldVisualization(linkage, omitBoundary=True)
    return v

def get_average_distance_to_target_surface(linkage):
    distance_to_surface = np.array(linkage.get_squared_distance_to_target_surface((linkage.visualizationGeometry()[0]).flatten()))
    distance_to_surface = np.sqrt(distance_to_surface)
    return np.sum(distance_to_surface)/len(distance_to_surface)

def get_max_distance_to_target_surface(linkage):
    distance_to_surface = np.array(linkage.get_squared_distance_to_target_surface((linkage.visualizationGeometry()[0]).flatten()))
    distance_to_surface = np.sqrt(distance_to_surface)
    return np.max(distance_to_surface)

def get_average_distance_to_target_joint(linkage, curved_save_tgt_joint_pos):
    jointPosDiff = linkage.jointPositions() - curved_save_tgt_joint_pos
    distances = [la.norm(jointPosDiff[3*x:3*(x+1)]) for x in range(linkage.numJoints())]
    return np.sum(distances)/linkage.numJoints()

# Visualization
def set_surface_view_options(linkage_view, linkage_color = 'lightgreen', surface_color = 'gray', linkage_transparent = False, surface_transparent = True):
    linkage_view.viewOptions[linkage_view.ViewType.LINKAGE].color = linkage_color
    linkage_view.viewOptions[linkage_view.ViewType.LINKAGE].transparent = linkage_transparent
    linkage_view.viewOptions[linkage_view.ViewType.SURFACE].transparent = surface_transparent
    linkage_view.viewOptions[linkage_view.ViewType.SURFACE].color = surface_color
    linkage_view.applyViewOptions()

def get_double_side_view(linkage, topColor = [0.5, 0.5, 0.5], bottomColor = [79/255., 158/255., 246/255.], flip = False):
    import linkage_vis
    if flip:
        topColor, bottomColor = bottomColor, topColor
    double_view = linkage_vis.getColoredRodOrientationViewer(linkage, width = 1024, height = 640, bottomColor=bottomColor, topColor=topColor)
    return double_view

def highlight_rod_and_joint(linkage, strip_index, select_joint_index, cross_section):
    import linkage_vis
    ribbons = order_segments_by_ribbons(linkage)
    new_rod, fixedVars = construct_elastic_rod_loop_from_rod_segments(linkage, ribbons[strip_index])
    # Set the material of the new rod to be the same as previously.
    new_rod.setMaterial(elastic_rods.RodMaterial('rectangle', 2000, 0.3, cross_section, stiffAxis=elastic_rods.StiffAxis.D1))
    single_rod_view_compare = linkage_vis.LinkageViewer(linkage, width=1024, height=640)
    single_rod_view = linkage_vis.LinkageViewer(new_rod, width=1024, height=640)

    j = linkage.joint(select_joint_index)
    seg_index = j.segments_A[0]
    vx_index = 0
    if linkage.segment(seg_index).startJoint != select_joint_index:
        vx_index = -1
#     joint_vector_field = [np.zeros_like(np.reshape(s.rod.gradient()[0:3*s.rod.numVertices()], (s.rod.numVertices(), 3))) for s in linkage.segments()]
#     joint_vector_field[seg_index][vx_index] = linkage.segment(seg_index).rod.deformedConfiguration().materialFrame[vx_index].d2
#     single_rod_view_compare.update(vectorField=joint_vector_field)

    sf = vis.fields.ScalarField(new_rod, 0.6 * np.ones_like(np.array(new_rod.deformedConfiguration().len)), colormap=cm.Blues, vmin = 0, vmax = 1)

    single_rod_view_compare.update(mesh = single_rod_view.mesh, preserveExisting=True, scalarField=sf)
    single_rod_view_compare.setCameraParams(((0.1380416750325228, 0.9648987923360046, 4.776431269112697),
     (0.9983340296894934, -0.054896765875897646, -0.01776260848808606),
     (0.0, 0.0, 0.0)))
    return single_rod_view_compare

def write_crossing_ribbon_info(linkage, filename, scale):
    ''' return the list of two ribbon indices per crossing and the list of crossing lists per ribbon
    '''
    ribbons = order_segments_by_ribbons(linkage)
    _, _, _, _, all_joint_index, _ = get_turning_angle_and_length_from_ordered_rods(ribbons, linkage, rest = True)
    all_joint_index_list = [j_list + [j_list[0]] for j_list in all_joint_index]

    def get_ribbon_crossing_list(index):
        selected_list = []
        selected_ribbon = []
        for ribbon_index, index_list in enumerate(all_joint_index_list):
            if index in set(index_list):
                selected_ribbon.append(ribbon_index)
                selected_list.append(index_list)
        # print("The crossing {} belongs to ribbon {}".format(index, ', '.join([str(x) for x in selected_ribbon])))
        # for i in range(len(selected_list)):
        #     print('Ribbon {}: {}'.format(selected_ribbon[i], selected_list[i]))
        return selected_ribbon
    pairs_of_ribbons_per_crossing = [get_ribbon_crossing_list(i) for i in range(linkage.numJoints())]

    with open(filename, 'w') as f:
        f.write('# crossing count {}, ribbon count {}\n'.format(linkage.numJoints(), len(ribbons)))
        for vx in linkage.jointPositions().reshape((linkage.numJoints(), 3)):
            vx *= scale
            f.write('v  {} {} {}\n'.format(vx[0], vx[1], vx[2]))
        for c_index, pair in enumerate(pairs_of_ribbons_per_crossing):
            if len(pair) == 1:
                f.write('crossing {}: {}\n'.format(c_index, pair[0]))
            else:
                if len(pair) != 2:
                    print("There shouldn't be more than two ribbons at a crossing!")
                f.write('crossing {}: {} {}\n'.format(c_index, pair[0], pair[1]))
        for r_index, index_list in enumerate(all_joint_index_list):
            f.write('ribbon {}: {}\n'.format(r_index, " ".join([str(x) for x in index_list])))

    return pairs_of_ribbons_per_crossing, all_joint_index_list

def write_crossing_ribbon_info_json(linkage, filename, scale):
    ''' return the list of two ribbon indices per crossing and the list of crossing lists per ribbon
    '''
    ribbons = order_segments_by_ribbons(linkage)
    _, _, _, _, all_joint_index, _ = get_turning_angle_and_length_from_ordered_rods(ribbons, linkage, rest = True)
    all_joint_index_list = [j_list + [j_list[0]] for j_list in all_joint_index]

    def get_crossing_json(pos, ribbon_pair):
        return {'position': pos, 'ribbons': ribbon_pair}

    def get_ribbon_json(crossing_index_list):
        return {'crossings': list(crossing_index_list)}

    def get_ribbon_crossing_list(index):
        selected_list = []
        selected_ribbon = []
        for ribbon_index, index_list in enumerate(all_joint_index_list):
            if index in set(index_list):
                selected_ribbon.append(ribbon_index)
                selected_list.append(index_list)

        return selected_ribbon
    pairs_of_ribbons_per_crossing = [get_ribbon_crossing_list(i) for i in range(linkage.numJoints())]
    crossing_positions = linkage.jointPositions().reshape((linkage.numJoints(), 3)) * scale

    crossing_info_list = [get_crossing_json(crossing_positions[c_index], pairs_of_ribbons_per_crossing[c_index]) for c_index in range(len(pairs_of_ribbons_per_crossing))]

    ribbon_info_list = [get_ribbon_json(all_joint_index_list[index]) for index in range(len(all_joint_index_list))]

    crossing_ribbon_info = {'crossings': crossing_info_list, 'ribbons': ribbon_info_list}

    with open(filename, 'w') as f:
        json.dump(crossing_ribbon_info, f, indent=4)
    return crossing_ribbon_info

def flip_joint_normal_sign(linkage, origin, neighbor):
    # Flip the terminal edge normal sign at origin with respect to the argument neighbor. 
    segment_index = 0
    curr_joint = linkage.joint(origin)
    connect_si = curr_joint.connectingSegment(neighbor)
    curr_joint.set_terminalEdgeNormalSign(connect_si, curr_joint.terminalEdgeNormalSign(connect_si) * -1)
    
def set_design_parameters_from_topology(curved_linkage, io):
    def normalize(vec):
        return vec / la.norm(vec)
    all_pts = curved_linkage.get_closest_point_for_visualization(np.array(curved_linkage.deformedPoints()).flatten())
    all_normals = np.array(curved_linkage.get_closest_point_normal(all_pts))
    all_pts = np.reshape(all_pts, (curved_linkage.numSegments(), io.SUBDIVISION_RESOLUTION + 1, 3))
    all_normals = np.reshape(all_normals, (curved_linkage.numSegments(), io.SUBDIVISION_RESOLUTION + 1, 3))
    def set_segment_rest_kappa(curved_linkage, i):
        rod = curved_linkage.segment(i).rod
        pts = all_pts[i]
        normals = all_normals[i]
        # Flip normal by directors.
        dc = rod.deformedConfiguration()
        directors = []
        directors.append(dc.materialFrame[0].d2)
        directors.extend([normalize(dc.materialFrame[edge_index - 1].d2 + dc.materialFrame[edge_index].d2) for edge_index in range(io.SUBDIVISION_RESOLUTION)[1:]])
        directors.append(dc.materialFrame[-1].d2)
        normals = np.array([normals[i] if np.dot(normals[i], directors[i]) > 0 else normals[i] * -1 for i in range(len(normals))])
        edges = np.array([pts[i+1] - pts[i] for i in range(len(pts)-1)])

        def get_rest_kappa(e1, e2, pn):
            cb = 2 * np.cross(e1, e2) / (la.norm(e1) * la.norm(e2) + np.dot(e1, e2))
            return np.dot(cb, pn)

        kappas = np.array([get_rest_kappa(edges[i], edges[i+1], normals[i+1]) for i in range(len(edges) - 1)])
        current_rk = rod.restKappas()

        for j in range(len(current_rk) - 2):
            current_rk[j+1][0] = kappas[j]
        curved_linkage.segment(i).rod.setRestKappas(current_rk)


    # Set rest length.
    segment_rest_length = []
    for i in range(curved_linkage.numSegments()):
        rod = curved_linkage.segment(i).rod
        pts = curved_linkage.get_closest_point_for_visualization(np.array(rod.deformedPoints()).flatten())
        pts = np.reshape(pts, (int(len(pts)/3), 3))
        rls = np.array([la.norm(pts[i+1] - pts[i]) for i in range(len(rod.deformedPoints())-1)])
        curved_linkage.segment(i).rod.setRestLengths(rls)
    #     because the segment rest length only include from joint to joint (so half of the first and last edge length is not considered in the segment length)
        segment_rest_length.append(sum(rls) - 0.5 * (rls[0] + rls[-1]))
    curved_linkage.setPerSegmentRestLength(segment_rest_length)
    # Set rest kappa.
    for i in range(curved_linkage.numSegments()):
        set_segment_rest_kappa(curved_linkage, i) 

def get_gravity_forces(linkage, io):
    ''' The return value can be used as the external force parameter in computing equilibrium. 
    '''
    external_forces = np.zeros(linkage.numDoF())
    centerline_index = np.array([[linkage.dofOffsetForCenterlinePos(i) + j for j in range(3)[2:]] for i in range(linkage.numCenterlinePos())]).flatten()
    sim_phy_scale = 1
    total_mass = 1e-10
    volume = io.RIBBON_CS[0] * io.RIBBON_CS[1] * linkage.totalRestLength() / linkage.numCenterlinePos()
    density = 1200 * 1e-12
    external_forces[centerline_index] = - volume * density * 9810
    return external_forces