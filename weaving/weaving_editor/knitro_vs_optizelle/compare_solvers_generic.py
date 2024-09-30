""" Generic script for optimizing sphere, torus, ellipsoid """
import os
import os.path as osp
elastic_rods_dir = '../../../elastic_rods/python/'
weaving_dir = '../../'
import sys
sys.path.append(elastic_rods_dir)
sys.path.append(weaving_dir)
import numpy as np
import elastic_rods
import linkage_vis
from bending_validation import suppress_stdout as so
from elastic_rods import EnergyType, InterleavingType
import linkage_optimization
import ribbon_linkage_helper as rlh

from function_helpers import initialize_linkage, get_linkage_eqm,\
                             compare_solvers, compare_energies,\
                             signed_diff, abs_diff, rel_diff


# Input parameters

# from conf_sphere import MODEL_NAME, MODEL_PATH, SUBDIVISION_RESOLUTION, INPUT_SURFACE_PATH, \
#                         RIBBON_CS, width, thickness, minRestLen, rw, sw, \
#                         optzel_trust_radius, optzel_mu
from conf_torus import MODEL_NAME, MODEL_PATH, SUBDIVISION_RESOLUTION, INPUT_SURFACE_PATH, \
                       RIBBON_CS, width, thickness, minRestLen, rw, sw, \
                       optzel_trust_radius, optzel_mu
# from conf_ellipse import MODEL_NAME, MODEL_PATH, SUBDIVISION_RESOLUTION, INPUT_SURFACE_PATH, \
#                          RIBBON_CS, width, thickness, minRestLen, rw, sw, \
#                          optzel_trust_radius, optzel_mu


USE_RESTLEN = True
SECOND_OPT = False

if USE_RESTLEN is False:
    minRestLen *= - 1


# Input object
default_camera_parameters = ((3.466009282140468, -4.674139805388271, -2.556131049738206),
                             (-0.21402574298422497, -0.06407538766530313, -0.9747681088523519),
                             (0.1111, 0.1865, 0.5316))

# Optimizer Parameters

OPTS = elastic_rods.NewtonOptimizerOptions()
OPTS.gradTol = 1e-6
OPTS.verbose = 10
OPTS.beta = 1e-8
OPTS.niter = 100
OPTS.verboseNonPosDef = False


# optizelle tolerance
knitro_tol = 1e-2
optzel_tol = 1e-2

basename = "{}_w{:5.3f}_t{:5.3f}_{}_{}".format(MODEL_NAME, width, thickness, rw, sw)
knitro_export = "knitro_{}".format(basename)
optzel_export = "optzel_{}".format(basename)

# Ribbon Optimization
# Initialize the linkage. Currently there are two types of design parameters: rest lengths and rest curvatures. The user can choose to activate or deactivate each one.

with so():
    curved_linkage = initialize_linkage(surface_path=INPUT_SURFACE_PATH,
                                        useCenterline=True,
                                        model_path=MODEL_PATH,
                                        cross_section=RIBBON_CS,
                                        subdivision_res=SUBDIVISION_RESOLUTION)
curved_linkage.set_design_parameter_config(use_restLen=True, use_restKappa=True)
curved_save_tgt_joint_pos = curved_linkage.jointPositions()
# curved_linkage_view = linkage_vis.LinkageViewer(curved_linkage)
# curved_linkage_view.show()

with so():
    curved_linkage2 = initialize_linkage(surface_path=INPUT_SURFACE_PATH,
                                         useCenterline=True,
                                         model_path=MODEL_PATH,
                                         cross_section=RIBBON_CS,
                                         subdivision_res=SUBDIVISION_RESOLUTION)
curved_linkage2.set_design_parameter_config(use_restLen=True, use_restKappa=True)
curved_save_tgt_joint_pos2 = curved_linkage2.jointPositions()
# curved_linkage_view2 = linkage_vis.LinkageViewer(curved_linkage2)
# curved_linkage_view2.show()

#
# Stage 1
#


# Callback function for updating the viewer during design parameter solve (Stage 1)
def curved_callback(prob, i):
    if (i % 20) != 0: return
    # curved_linkage_view.update()


def curved_callback2(prob, i):
    if (i % 20) != 0: return
    # curved_linkage_view2.update()


curved_dpo = elastic_rods.get_designParameter_optimizer(curved_linkage, rw, sw, callback=curved_callback)
curved_dpo.options.niter = 10000
curved_dpp = curved_dpo.get_problem()

curved_dpo2 = elastic_rods.get_designParameter_optimizer(curved_linkage2, rw, sw, callback=curved_callback2)
curved_dpo2.options.niter = 10000
curved_dpp2 = curved_dpo2.get_problem()


# Because how the weights work, this following line should not be run a second time.

with so():
    curved_cr = curved_dpo.optimize()
    curved_cr2 = curved_dpo2.optimize()

# Equilibrium solve

# with so():
elastic_rods.compute_equilibrium(curved_linkage, options=OPTS)
elastic_rods.compute_equilibrium(curved_linkage2, options=OPTS)

#with so():
# curved_linkage_view.update()
# curved_linkage_view2.update()

#
# Stage 2
#

# `linkage_optimization` is defined in `python_bindings/linkage_optimization.cc`. The `LinkageOptimization.cc .hh` files are for the XShell (the naming is confusing but this is the unmerged stage. We will unify this after the new solver is in place.

#
# Initialize the optimizers
#

# Optimizer for Knitro
OPTS.niter = 25
useCenterline = True
curved_optimizer = linkage_optimization.WeavingOptimization(curved_linkage,
                                                            INPUT_SURFACE_PATH,
                                                            useCenterline,
                                                            equilibrium_options=OPTS,
                                                            pinJoint=0,
                                                            useFixedJoint=False)
curved_optimizer.set_target_joint_position(curved_save_tgt_joint_pos)
# curved_linkage_view.update()

# Optimizer copy for Optizelle
curved_optimizer2 = linkage_optimization.WeavingOptimization(curved_linkage2,
                                                             INPUT_SURFACE_PATH,
                                                             useCenterline,
                                                             equilibrium_options=OPTS,
                                                             pinJoint=0,
                                                             useFixedJoint=False)
curved_optimizer2.set_target_joint_position(curved_save_tgt_joint_pos2)
# curved_linkage_view2.update()
OPTS.niter = 200

curved_optimizer.rl_regularization_weight = rw
curved_optimizer.smoothing_weight = sw
curved_optimizer.beta = 500000.0
curved_optimizer.gamma = 1

curved_optimizer2.rl_regularization_weight = rw
curved_optimizer2.smoothing_weight = sw     # 10 is good
curved_optimizer2.beta = 500000.0
curved_optimizer2.gamma = 1

algorithm = linkage_optimization.WeavingOptAlgorithm.NEWTON_CG


def curved_update_viewer():
    pass
    # curved_linkage_view.update()


# We need to run the stage 2 optimizer first with high surface attraction weight.

# Define a few params

trust_region_radius = 1.0
tolerance = knitro_tol

#
# 1st Optimization
#

# Optimize with Knitro

itermax = 2000
curved_optimizer.WeavingOptimize(algorithm, itermax, trust_region_radius, tolerance, curved_update_viewer, minRestLen)

# Optimize with Optizelle

itermax = 550
trust_region_radius = optzel_trust_radius
intpoint_barrier_height = optzel_mu
tolerance = optzel_tol  # Optizelle needs a smaller tolerance for giving the same results
curved_optimizer2.WeavingOptimizeOptizelle(algorithm, itermax, trust_region_radius, tolerance, intpoint_barrier_height, minRestLen)

# Compare 1st results between KNITRO and OPTIZELLE
# Extract solutions and objective functions values

param_knitro, obj_knitro = np.asarray(curved_optimizer.m_final_params), curved_optimizer.m_final_objective
param_optzel, obj_optzel = np.asarray(curved_optimizer2.m_final_params), curved_optimizer2.m_final_objective
grad_knitro = np.asarray(curved_optimizer.m_final_grad)
grad_optzel = np.asarray(curved_optimizer2.m_final_grad)

# curved_linkage_view.update()
# curved_linkage_view2.update()

compare_solvers(param_knitro, param_optzel, obj_knitro, obj_optzel, grad_knitro, grad_optzel,
                param_tol=1e-1, obj_tol=1e-7, grad_tol=1e-1, fname=basename, silent=True)

if SECOND_OPT:
    #
    # 2nd Optimization
    #

    # Then we lower this weight and allow the closest point projections to change.

    # Knitro
    curved_optimizer.setLinkageAttractionWeight(1e-5)
    curved_optimizer.set_holdClosestPointsFixed(False)

    # Optizelle
    curved_optimizer2.setLinkageAttractionWeight(1e-5)
    curved_optimizer2.set_holdClosestPointsFixed(False)

    trust_region_radius = 1.0
    tolerance = knitro_tol

    # Knitro
    itermax = 2000
    curved_optimizer.WeavingOptimize(algorithm, itermax, trust_region_radius, tolerance, curved_update_viewer, minRestLen)

    # Optizelle
    itermax = 250
    trust_region_radius = optzel_trust_radius
    intpoint_barrier_height = optzel_mu
    tolerance = optzel_tol      # Optizelle needs a smaller tolerance for giving the same results
    curved_optimizer2.WeavingOptimizeOptizelle(algorithm, itermax, trust_region_radius, tolerance, intpoint_barrier_height, minRestLen)

    # Compare 2nd results between KNITRO and OPTIZELLE

    # Extract solutions and objective functions values
    param_knitro, obj_knitro = np.asarray(curved_optimizer.m_final_params), curved_optimizer.m_final_objective
    param_optzel, obj_optzel = np.asarray(curved_optimizer2.m_final_params), curved_optimizer2.m_final_objective
    grad_knitro = np.asarray(curved_optimizer.m_final_grad)
    grad_optzel = np.asarray(curved_optimizer2.m_final_grad)

    compare_solvers(param_knitro, param_optzel, obj_knitro, obj_optzel, grad_knitro, grad_optzel, param_tol=1e-3, obj_tol=1e-9, grad_tol=1e-3, fname=basename)


# Validate whether the surface attraction weight is set low enough.

curved_optimizer_energy = curved_linkage.energy()
validation_curved_linkage = curved_optimizer.getLinesearchWeaverLinkage()
validation_curved_linkage.attraction_weight = 1e-7
with so():
    elastic_rods.compute_equilibrium(validation_curved_linkage, options=OPTS)
# validation_curved_view = linkage_vis.LinkageViewer(validation_curved_linkage, width=1024, height=640)
validation_curved_energy = validation_curved_linkage.energy()


curved_optimizer_energy2 = curved_linkage2.energy()
validation_curved_linkage2 = curved_optimizer2.getLinesearchWeaverLinkage()
validation_curved_linkage2.attraction_weight = 1e-7
with so():
    elastic_rods.compute_equilibrium(validation_curved_linkage2, options=OPTS)
# validation_curved_view = linkage_vis.LinkageViewer(validation_curved_linkage2, width=1024, height=640)
validation_curved_energy2 = validation_curved_linkage2.energy()

# Output energies and differences between solvers

linkages = [curved_linkage, curved_linkage2]
validation_energies = [validation_curved_energy, validation_curved_energy2]

compare_energies(linkages, validation_energies, filename=basename, obj=[obj_knitro, obj_optzel])

# Export .obj

rlh.export_linkage_geometry_to_obj(curved_linkage, knitro_export)
rlh.export_linkage_geometry_to_obj(curved_linkage2, optzel_export)
# curved_linkage_view.update()
# curved_linkage_view2.update()
