#!/usr/bin/env python
elastic_rods_dir = '../../elastic_rods/python/'
weaving_dir = '../'
import os
import os.path as osp
import sys; sys.path.append(elastic_rods_dir); sys.path.append(weaving_dir)
import numpy as np, elastic_rods
import numpy.linalg as la
from elastic_rods import EnergyType, InterleavingType

# weaving
from pipeline_helper import (initialize_linkage, stage_1_optimization, initialize_stage_2_optimizer, stage_2_optimization, InputOrganizer, set_surface_view_options)
import vis.fields

import parallelism
parallelism.set_max_num_tbb_threads(12)
parallelism.set_hessian_assembly_num_threads(4)
parallelism.set_gradient_assembly_num_threads(4)

rod_length = 0.3534025445286393
width = rod_length / 15 * 5
thickness = width / 5 * 0.35
print(width, thickness)
scale = 1
io = InputOrganizer('sphere_{}'.format(scale), thickness, width, weaving_dir)

import py_newton_optimizer
OPTS = py_newton_optimizer.NewtonOptimizerOptions()
OPTS.gradTol = 1e-8
OPTS.verbose = 1;
OPTS.beta = 1e-8
OPTS.niter = 200
OPTS.verboseNonPosDef = False
rw = 0.01
sw = 10
drw = 0.001
dsw = 0.01

def test_sphere():
    curved_linkage = initialize_linkage(surface_path = io.SURFACE_PATH, useCenterline = True, model_path = io.MODEL_PATH, cross_section = io.RIBBON_CS, subdivision_res = io.SUBDIVISION_RESOLUTION)
    curved_linkage.set_design_parameter_config(use_restLen = True, use_restKappa = True)
    curved_save_tgt_joint_pos = curved_linkage.jointPositions();

    iterateData = stage_1_optimization(curved_linkage, rw, sw, None)

    curved_linkage.attraction_weight = 1e-16


    def eqm_callback(prob, i):
        return


    elastic_rods.compute_equilibrium(curved_linkage, callback = eqm_callback, options = OPTS)

    optimizer = initialize_stage_2_optimizer(curved_linkage, io.SURFACE_PATH, curved_save_tgt_joint_pos, None, rw, sw)

    optimizer, opt_iterateData, _ = stage_2_optimization(optimizer, curved_linkage, io.SURFACE_PATH, curved_save_tgt_joint_pos, None, -1, [], -5, 3)


if __name__ == "__main__":
    test_sphere()