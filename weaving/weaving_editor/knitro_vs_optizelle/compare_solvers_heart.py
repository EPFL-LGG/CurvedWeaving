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


# ## Input parameters
rod_length = 0.3534025445286393
width = rod_length / 15 * 4.0
thickness = width / 5 * 0.35 * 1.2
RIBBON_CS = [thickness, width]
minRestLen = RIBBON_CS[1] * 2.0 * 0.2

# Sphere 1
default_camera_parameters = ((3.466009282140468, -4.674139805388271, -2.556131049738206),
                             (-0.21402574298422497, -0.06407538766530313, -0.9747681088523519),
                             (0.1111, 0.1865, 0.5316))
MODEL_NAME = "heart_coarse_1"
MODEL_PATH = osp.join(weaving_dir + 'normalized_objs/models/{}.obj'.format(MODEL_NAME))
SUBDIVISION_RESOLUTION = 20
INPUT_SURFACE_PATH = osp.join(weaving_dir + 'normalized_objs/surface_models/{}.obj'.format(MODEL_NAME))

# Optimizer Parameters

OPTS = elastic_rods.NewtonOptimizerOptions()
OPTS.gradTol = 1e-6
OPTS.verbose = 10
OPTS.beta = 1e-8
OPTS.niter = 100
OPTS.verboseNonPosDef = False

rw = 0.1 * 20
sw = 0.1

#
# Helper functions
#


def initialize_linkage(surface_path=INPUT_SURFACE_PATH,
                       useCenterline=True,
                       cross_section=RIBBON_CS,
                       subdivision_res=SUBDIVISION_RESOLUTION,
                       model_path=MODEL_PATH):
    """ Initialize Woven Linkage

    There are two different types of linkage:
    the `RodLinkage` and the `SurfaceAttractedLinkage`. The `SurfaceAttractedLinkage` has the additional surface attraction weight and hence need the `surface_path` as parameter.
    """
    l = elastic_rods.SurfaceAttractedLinkage(surface_path,
                                             useCenterline,
                                             model_path,
                                             subdivision_res,
                                             False,
                                             InterleavingType.triaxialWeave)
    l.setMaterial(elastic_rods.RodMaterial('rectangle', 200, 0.3, cross_section, stiffAxis=elastic_rods.StiffAxis.D1))
    l.set_holdClosestPointsFixed(True)
    l.set_attraction_tgt_joint_weight(0.01)
    l.attraction_weight = 100
    return l


def get_linkage_eqm(l, opt, cam_param=default_camera_parameters, target_surf=None):
    """ Compute Equilibrium
        Similarly there are two different types of viewer depending on whether the surface is visualized.
    """
    elastic_rods.compute_equilibrium(l, options=opt)
    # if (target_surf is None):
    #     view = linkage_vis.LinkageViewer(l, width=1024, height=640)
    # else:
    #     view = linkage_vis.LinkageViewerWithSurface(l, target_surf, width=1024, height=640)
    # view.setCameraParams(cam_param)
    # return l, view
    return l, None


def compare_solvers(pk, po, ok, oo, param_tol=1e-4, obj_tol=1e-4):
    """ Compare Knitro and Optizelle results """
    p_diff = abs(pk - po)
    o_diff = abs(oo - ok)
    suspicious = 0
    for i,item in enumerate(p_diff):
        if item > param_tol:
            suspicious = 1
            print("Significant difference for parameter:", i, item, pk[i], po[i])
    if o_diff > obj_tol:
        suspicious = 1
        print("Significant difference for objective values:", ok, oo, o_diff)
    if not suspicious:
        print("No significant differences for parameters and objective values")


# Curved Ribbon Optimization
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
    if (i % 20) != 0:
        return
    return
    # curved_linkage_view.update()

def curved_callback2(prob, i):
    if (i % 20) != 0:
        return
    return
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

curved_optimizer.rl_regularization_weight = 0.1
curved_optimizer.smoothing_weight = 10
curved_optimizer.beta = 500000.0
curved_optimizer.gamma = 1

curved_optimizer2.rl_regularization_weight = 0.1
curved_optimizer2.smoothing_weight = 10
curved_optimizer2.beta = 500000.0
curved_optimizer2.gamma = 1

algorithm = linkage_optimization.WeavingOptAlgorithm.NEWTON_CG


def curved_update_viewer():
    pass
    # curved_linkage_view.update()


# We need to run the stage 2 optimizer first with high surface attraction weight.

# Define a few params

trust_region_radius = 1.0
tolerance = 1e-2

#
# 1st Optimization
#

# Optimize with Knitro

itermax = 2000
with so():
    curved_optimizer.WeavingOptimize(algorithm, itermax, trust_region_radius, tolerance, curved_update_viewer, minRestLen)

# Optimize with Optizelle

itermax = 1000
intpoint_barrier_height = 0.9
tolerance /= 1e3    # Optizelle needs a smaller tolerance for giving the same results
curved_optimizer2.WeavingOptimizeOptizelle(algorithm, itermax, trust_region_radius, tolerance, intpoint_barrier_height, minRestLen)

# Compare 1st results between KNITRO and OPTIZELLE
# Extract solutions and objective functions values

param_knitro, obj_knitro = np.asarray(curved_optimizer.m_final_params), curved_optimizer.m_final_objective
param_optizelle, obj_optizelle = np.asarray(curved_optimizer2.m_final_params), curved_optimizer2.m_final_objective

# curved_linkage_view.update()
# curved_linkage_view2.update()

compare_solvers(param_knitro, param_optizelle, obj_knitro, obj_optizelle, param_tol=1e-3, obj_tol=1e-5)

# #
# # 2nd Optimization
# #

# # Then we lower this weight and allow the closest point projections to change.

# # Knitro
# curved_optimizer.setLinkageAttractionWeight(1e-5)
# curved_optimizer.set_holdClosestPointsFixed(False)

# # Optizelle
# curved_optimizer2.setLinkageAttractionWeight(1e-5)
# curved_optimizer2.set_holdClosestPointsFixed(False)

# trust_region_radius = 1.0
# tolerance = 1e-2

# # Knitro
# itermax = 2000
# curved_optimizer.WeavingOptimize(algorithm, itermax, trust_region_radius, tolerance, curved_update_viewer, minRestLen)

# # Optizelle
# itermax = 2000
# intpoint_barrier_height = 0.7
# tolerance /= 1e2       # Optizelle needs a smaller tolerance for giving the same results
# curved_optimizer2.WeavingOptimizeOptizelle(algorithm, itermax, trust_region_radius, tolerance, intpoint_barrier_height, minRestLen)


# # Compare 2nd results between KNITRO and OPTIZELLE

# # Extract solutions and objective functions values
# param_knitro, obj_knitro = np.asarray(curved_optimizer.m_final_params), curved_optimizer.m_final_objective
# param_optizelle, obj_optizelle = np.asarray(curved_optimizer2.m_final_params), curved_optimizer2.m_final_objective

# compare_solvers(param_knitro, param_optizelle, obj_knitro, obj_optizelle, param_tol=5e-3, obj_tol=1e-7)


# Validate whether the surface attraction weight is set low enough.

curved_optimizer_energy = curved_linkage.energy()
validation_curved_linkage = curved_optimizer.getLinesearchWeaverLinkage()
validation_curved_linkage.attraction_weight = 1e-7
with so():
    elastic_rods.compute_equilibrium(validation_curved_linkage, options=OPTS)
# validation_curved_view = linkage_vis.LinkageViewer(validation_curved_linkage, width=1024, height=640)
validation_curved_energy = validation_curved_linkage.energy()

print("knitro curved opt energy        ", curved_optimizer_energy)
print("knitro validation curved energy", validation_curved_energy)
print(abs((validation_curved_energy - curved_optimizer_energy) / curved_optimizer_energy))

curved_optimizer_energy2 = curved_linkage2.energy()
validation_curved_linkage2 = curved_optimizer2.getLinesearchWeaverLinkage()
validation_curved_linkage2.attraction_weight = 1e-7
with so():
    elastic_rods.compute_equilibrium(validation_curved_linkage2, options=OPTS)
# validation_curved_view = linkage_vis.LinkageViewer(validation_curved_linkage2, width=1024, height=640)
validation_curved_energy2 = validation_curved_linkage2.energy()

print("optizelle curved opt energy       ", curved_optimizer_energy2)
print("optizelle validation curved energy", validation_curved_energy2)
print(abs((validation_curved_energy2 - curved_optimizer_energy2) / curved_optimizer_energy2))

print("optizelle curved opt energy - knitro curved opt energy", curved_optimizer_energy2 - curved_optimizer_energy)

knitro_export    = "knitro_w{:5.3f}_t{:5.3f}_{}_{}".format(width, thickness, rw, sw)
optizelle_export = "optzel_w{:5.3f}_t{:5.3f}_{}_{}".format(width, thickness, rw, sw)
rlh.export_linkage_geometry_to_obj(curved_linkage, knitro_export)
rlh.export_linkage_geometry_to_obj(curved_linkage2, optizelle_export)

# curved_linkage_view.update()
# curved_linkage_view2.update()
