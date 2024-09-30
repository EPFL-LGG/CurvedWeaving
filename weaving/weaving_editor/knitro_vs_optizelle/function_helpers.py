""" Contains all the functions needed for the scripts for comparing solvers. """
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


# ------------------------------------------------------------------------------------------------------------------------
# LINKAGE
# ------------------------------------------------------------------------------------------------------------------------


def initialize_linkage(surface_path, useCenterline, cross_section, subdivision_res, model_path):
    """
    Initialize Woven Linkage

    There are two different types of linkage:
    the `RodLinkage` and the `SurfaceAttractedLinkage`. The `SurfaceAttractedLinkage` has the additional surface attraction weight and hence need the `surface_path` as parameter.
    """
    l = elastic_rods.SurfaceAttractedLinkage(surface_path,
                                             useCenterline,
                                             model_path,
                                             subdivision_res,
                                             False,
                                             InterleavingType.triaxialWeave)
    l.setMaterial(elastic_rods.RodMaterial('rectangle', 2000, 0.3, cross_section, stiffAxis=elastic_rods.StiffAxis.D1))
    l.set_holdClosestPointsFixed(True)
    l.set_attraction_tgt_joint_weight(0.01)
    l.attraction_weight = 100
    return l


def get_linkage_eqm(l, opt, cam_param, target_surf=None):
    """
    Compute Equilibrium

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

# ------------------------------------------------------------------------------------------------------------------------
# COMPARE SOLVERS
# ------------------------------------------------------------------------------------------------------------------------


def compare_solvers(pk, po, ok, oo, gk, go, param_tol=1e-4, obj_tol=1e-4, grad_tol=1e-4, fname="", silent=False):
    """ Compare Knitro and Optizelle results """
    print("Start of param compare")
    param_fname = "{}_diff_param".format(fname)
    grad_fname = "{}_diff_grad".format(fname)
    # Last value of param is actually the objective function's value
    ook, ooo = np.array([ok]), np.array([oo])
    pk = np.append(pk, ook)
    po = np.append(po, ooo)
    suspicious = compare_vectors(pk, po, tol=param_tol, filename=param_fname, silent=silent)
    print("End of param compare")
    print("Start of grad compare")
    suspicious = compare_vectors(gk, go, tol=grad_tol, filename=grad_fname, silent=silent)
    print("End of grad compare")
    o_diff = abs(oo - ok)
    if o_diff > obj_tol:
        suspicious = True
        print("Significant difference for objective values:", ok, oo, o_diff)
    if not suspicious:
        print("No significant differences.")


def compare_vectors(a, b, tol=1e-4, to_file=False, filename="vector_diff_out", silent=False):
    threshold_reached = False
    absdiff = abs_diff(a, b)
    #
    fname_full = "{}_full_diff.txt".format(filename)
    fname_diff = "{}_selected_diff.txt".format(filename)
    buf_full   = open(fname_full, 'w+')
    buf_diff   = open(fname_diff, 'w+')
    for buf in (buf_diff, buf_full):
        buf.write("#Elements significative diffs\n")
        buf.write("#i    knitro       optzel       abs diff     signed diff  rel diff\n")
        buf.write("#-----------------------------------------------------------------\n")

    if silent is False:
        print("Elements significative diffs")
        print("i     knitro       optzel       abs diff     signed diff  rel diff")
        print("------------------------------------------------------------------")

    for i, item in enumerate(absdiff):
        msg = "{:>5d} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e}".format(i, a[i], b[i], item,
                                                                                signed_diff(a[i], b[i]),
                                                                                rel_diff(a[i], b[i]))
        if item > tol:
            threshold_reached = True
            if silent is False:
                print(msg)
            buf_diff.write(msg + "\n")
        buf_full.write(msg + "\n")
    buf_diff.close()
    buf_full.close()
    return threshold_reached


def signed_diff(a, b):
    return a - b


def abs_diff(a, b):
    return abs(a - b)


def rel_diff(a, b):
    return abs(a - b) / a


def compare_energies(linkages, validation_energies, filename="energies_diff_out", obj=None):
    filename = "{}_energies.txt".format(filename)
    buf = open(filename, "w+")
    solver = ["knitro", "optzel"]
    if obj is not None:
        msg = "min J knitro and optzel {:>12.4e} {:>12.4e} {:>12.4e} {:>12.4e}\n".format(obj[0], obj[1],
                                                                                         signed_diff(obj[0], obj[1]),
                                                                                         rel_diff(obj[0], obj[1]))
        print(msg)
        buf.write(msg)
    for i, l in enumerate(linkages):
        msg  = "\n"
        msg += "                               E            Es           Eb           Et\n"
        msg += "{} curved opt energy    {:>12.3e} {:>12.3e} {:>12.3e} {:>12.3e}\n".format(solver[i],
                                                                                          l.energy(),
                                                                                          l.energyStretch(),
                                                                                          l.energyBend(),
                                                                                          l.energyTwist())
        print(msg)
        buf.write(msg)
        msg  = "{} Elastic Energy      {:>12.3e}\n".format(solver[i], l.energyBend() + l.energyStretch() + l.energyTwist())
        msg += "Validation curved energy    {:>12.3e}\n".format(validation_energies[i])
        msg += "abs and relative diff       {:>12.3e} {:>12.3e}\n".format(abs_diff(l.energy(), validation_energies[i]),
                                                                                              rel_diff(l.energy(), validation_energies[i]))
        print(msg)
        buf.write(msg + "\n")


    msg  = "Optizelle -  Knitro energy                      {:>12.3e}\n".format(linkages[1].energy() - linkages[0].energy())
    msg += "abs(Optizelle -  Knitro energy) / Knitro energy {:>12.3e}\n".format(rel_diff(linkages[1].energy(), linkages[0].energy()))
    print(msg)
    buf.write(msg)
    buf.close()