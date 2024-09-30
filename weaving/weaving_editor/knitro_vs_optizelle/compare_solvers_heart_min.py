import os
import time
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

import copy as cp


# Parameters are taken from heart_coarse_minimal.ipynb

# Set to True in the options below if you already optimized a specific linkage
# and you would like to reuse the optimized linkage. Loading is performed in
# place of the full design optimization.
# NOTE: Doesn't seem to work at the moment, leave at False
LOAD_OPTIMIZED_DOFS = False

# Set to False if strips cannot be labeled into families for a model
USE_FAMILY_LABEL = False

# ## Input parameters
width = 1 / 150
thickness = 1 / 15
RIBBON_CS = [thickness, width]
minRestLen = width

# Sphere 1
default_camera_parameters = ((3.466009282140468, -4.674139805388271, -2.556131049738206),
                             (-0.21402574298422497, -0.06407538766530313, -0.9747681088523519),
                             (0.1111, 0.1865, 0.5316))
MODEL_NAME = "heart_coarse_1"
MODEL_PATH = osp.join(weaving_dir, 'scaled_objs/models/{}.obj'.format(MODEL_NAME))
SUBDIVISION_RESOLUTION = 5
SMOOTHING_WEIGHT = 0.1
REGULARIZATION_WEIGHT = 0.1
INPUT_SURFACE_PATH = osp.join(weaving_dir, 'scaled_objs/surface_models/{}.obj'.format(MODEL_NAME))
RIBBON_NAME = "heart_coarse_1_strip"
# Optimizer Parameters

OPTS = elastic_rods.NewtonOptimizerOptions()
OPTS.gradTol = 1e-6
OPTS.verbose = 10
OPTS.beta = 1e-8
OPTS.niter = 100
OPTS.verboseNonPosDef = False

sw = SMOOTHING_WEIGHT
rw = REGULARIZATION_WEIGHT

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
    l = elastic_rods.SurfaceAttractedLinkage(surface_path, useCenterline, model_path, subdivision_res, False, InterleavingType.triaxialWeave)
    l.setMaterial(elastic_rods.RodMaterial('rectangle', 2000, 0.3, cross_section, stiffAxis=elastic_rods.StiffAxis.D1))
    l.setDoFs(l.getDoFs())
    l.set_holdClosestPointsFixed(True)
    l.set_attraction_tgt_joint_weight(0.01)
    l.attraction_weight = 100
    return l


def initialize_normal_linkage(cross_section=RIBBON_CS, subdivision_res=SUBDIVISION_RESOLUTION, model_path=MODEL_PATH):
    l = elastic_rods.RodLinkage(model_path, subdivision_res, False)
    l.setMaterial(elastic_rods.RodMaterial('rectangle', 2000, 0.3, cross_section, stiffAxis=elastic_rods.StiffAxis.D1))
    return l


def design_parameter_solve(l,regularization_weight=0.1, smoothing_weight=1):
    design_opts = elastic_rods.NewtonOptimizerOptions()
    design_opts.niter = 10000
    design_opts.verbose = 10
    l.set_design_parameter_config(use_restLen=True, use_restKappa=True)
    elastic_rods.designParameter_solve(l, design_opts, regularization_weight=0.0, smoothing_weight=0.001)
    l.set_design_parameter_config(use_restLen=True, use_restKappa=True)


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


def rel_diff(a, b):
    return abs(a - b) / a


def update_viewer():
    pass


# Generate straight linkage

straight_linkage = initialize_linkage(surface_path=INPUT_SURFACE_PATH,
                                      useCenterline=True,
                                      model_path=MODEL_PATH,
                                      cross_section=RIBBON_CS,
                                      subdivision_res=SUBDIVISION_RESOLUTION)

straight_linkage.attraction_weight = 0
with so():
    straight_linkage, intial_view = get_linkage_eqm(straight_linkage, cam_param=default_camera_parameters, opt=OPTS)
straight_rod_dof = straight_linkage.getDoFs()

#
# Generate straight linkage equilibrium
#

with so():
    linkage = initialize_linkage(surface_path=INPUT_SURFACE_PATH,
                                 useCenterline=True,
                                 model_path=MODEL_PATH,
                                 cross_section=RIBBON_CS,
                                 subdivision_res=SUBDIVISION_RESOLUTION)
# with so(): linkage = initialize_normal_linkage(model_path = MODEL_PATH, cross_section = RIBBON_CS, subdivision_res = SUBDIVISION_RESOLUTION)
save_tgt_joint_pos = linkage.jointPositions()
with so():
    design_parameter_solve(linkage, regularization_weight=rw, smoothing_weight=sw)

link = [None, None]
optimizer = [None, None]
start_time = time.time()
with so():
    link[0], view = get_linkage_eqm(linkage, cam_param=default_camera_parameters, opt=OPTS, target_surf=INPUT_SURFACE_PATH)
    link[1], view = get_linkage_eqm(linkage, cam_param=default_camera_parameters, opt=OPTS, target_surf=INPUT_SURFACE_PATH)
print('\ncompute equilibrium takes: {}\n'.format(time.time() - start_time))


OPTS.niter = 200
useCenterline = True
algorithm = linkage_optimization.WeavingOptAlgorithm.NEWTON_CG
print("----> 1")
for i in range(0, 2):
    link[i].set_design_parameter_config(use_restLen=True, use_restKappa=True)
    optimizer[i] = linkage_optimization.WeavingOptimization(link[i],
                                                            INPUT_SURFACE_PATH,
                                                            useCenterline,
                                                            equilibrium_options=OPTS,
                                                            pinJoint=0,
                                                            useFixedJoint=False)
    optimizer[i].set_target_joint_position(save_tgt_joint_pos)

    optimizer[i].rl_regularization_weight = REGULARIZATION_WEIGHT
    optimizer[i].smoothing_weight = SMOOTHING_WEIGHT
    optimizer[i].beta = 500000.0
    optimizer[i].gamma = 1
    #
    itermax = 1000
    trust_region_radius = 1.0
    intpoint_barrier_height = 0.4
    tolerance = [1e-1, 1e-1]

    if not LOAD_OPTIMIZED_DOFS:
        if i == 0:
            optimizer[i].WeavingOptimize(algorithm, 2000, trust_region_radius, tolerance[0], update_viewer)
        else:
            optimizer[i].WeavingOptimizeOptizelle(algorithm, itermax, trust_region_radius, tolerance[1], intpoint_barrier_height, RIBBON_CS[1])
    else:
        loadedDoFs = np.load("dof_files/{}_{}.npy".format(MODEL_NAME, i))
        link[i].setExtendedDoFs(loadedDoFs)

    # Store DoFs to file
    if not os.path.exists("dof_files"):
        os.makedirs("dof_files")
    rawDoFs = link[i].getExtendedDoFs()
    np.save("dof_files/{}_{}.npy".format(MODEL_NAME, i), rawDoFs)

for i in range(0, 2):
    #link[i].energy(elastic_rods.SurfaceAttractionEnergyType.Elastic)
    optimizer[i].setLinkageAttractionWeight(1e-16)
    optimizer[i].set_holdClosestPointsFixed(False)

    if not LOAD_OPTIMIZED_DOFS:
        if i == 0:
            optimizer[i].WeavingOptimize(algorithm, 2000, trust_region_radius, tolerance[i], update_viewer)
        else:
            optimizer[i].WeavingOptimizeOptizelle(algorithm, itermax, trust_region_radius, tolerance[i], intpoint_barrier_height, RIBBON_CS[1])
    else:
        loadedDoFs = np.load("dof_files/{}_{}.npy".format(MODEL_NAME, i))
        linkage.setExtendedDoFs(loadedDoFs)

linesearch_linkage = [o.getLinesearchWeaverLinkage() for o in optimizer]
optimized_energy = [item.energy(elastic_rods.SurfaceAttractionEnergyType.Elastic) for item in link]

start_time = time.time()
validation_linkage = [None, None]
for i in range(0, 2):
    with so():
        validation_linkage[i], validation_view = get_linkage_eqm(linesearch_linkage[i],
                                                                 cam_param=default_camera_parameters,
                                                                 opt=OPTS,
                                                                 target_surf=INPUT_SURFACE_PATH)
print('\ncompute equilibrium takes: {}\n'.format(time.time() - start_time))
validation_energy = [vl.energy(elastic_rods.SurfaceAttractionEnergyType.Elastic) for vl in validation_linkage]

# Validate whether the surface attraction weight is set low enough.

print("knitro opt energy               ", optimized_energy[0])
print("knitro validation curved energy ", validation_energy[0])
print("relative difference             ", rel_diff(optimized_energy[0], validation_energy[0]))

print("optzel opt energy               ", optimized_energy[1])
print("otpzel validation curved energy ", validation_energy[1])
print("relative difference             ", rel_diff(optimized_energy[1], validation_energy[1]))
print("opt energy - knitro energy (abs and rel)", optimized_energy[1] - optimized_energy[0], rel_diff(*optimized_energy))

knitro_export_ol = "knitro_heart_ol_w{:5.3f}_t{:5.3f}_{}_{}".format(width, thickness, rw, sw)
knitro_export_vl = "knitro_heart_vl_w{:5.3f}_t{:5.3f}_{}_{}".format(width, thickness, rw, sw)
knitro_export_sl = "knitro_heart_sl_w{:5.3f}_t{:5.3f}_{}_{}".format(width, thickness, rw, sw)
rlh.export_linkage_geometry_to_obj(link[0], knitro_export_ol)
rlh.export_linkage_geometry_to_obj(validation_linkage[0], knitro_export_vl)
rlh.export_linkage_geometry_to_obj(straight_linkage, knitro_export_sl)

optzel_export_ol = "optzel_heart_ol_w{:5.3f}_t{:5.3f}_{}_{}".format(width, thickness, rw, sw)
optzel_export_vl = "optzel_heart_vl_w{:5.3f}_t{:5.3f}_{}_{}".format(width, thickness, rw, sw)
optzel_export_sl = "optzel_heart_sl_w{:5.3f}_t{:5.3f}_{}_{}".format(width, thickness, rw, sw)
rlh.export_linkage_geometry_to_obj(link[1], optzel_export_ol)
rlh.export_linkage_geometry_to_obj(validation_linkage[1], optzel_export_vl)
rlh.export_linkage_geometry_to_obj(straight_linkage, optzel_export_sl)
