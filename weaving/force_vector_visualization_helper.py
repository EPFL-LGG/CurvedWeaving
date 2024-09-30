elastic_rods_dir = '../elastic_rods/python/'
weaving_dir = './'
import os
import os.path as osp
import sys; sys.path.append(elastic_rods_dir); sys.path.append(weaving_dir)
import numpy as np, elastic_rods, linkage_vis
import numpy.linalg as la
from elastic_rods import EnergyType, InterleavingType
import py_newton_optimizer
OPTS = py_newton_optimizer.NewtonOptimizerOptions()
OPTS.gradTol = 1e-8
OPTS.verbose = 1;
OPTS.beta = 1e-8
OPTS.niter = 200
OPTS.verboseNonPosDef = False
rw = 1
sw = 10
drw = 0.1
dsw = 0.1
import pickle 
import gzip
import matplotlib.pyplot as plt

class ForceVector:
    '''
    A data class for gathering information needed for rendering force vectors on linkages. 
    '''
    def __init__(self, joint_pos, force_vector):
        self.joint_pos = joint_pos
        self.magnitude = la.norm(force_vector)
        self.direction = force_vector / self.magnitude
        self.color = None

def get_color_scheme(colors):
    '''
    Color scheme for force vectors.
    '''
#         cmap = plt.cm.plasma
    cmap = plt.cm.PuRd
    return cmap(colors)
#     import proplot as plot

#     # Colormap from named color
#     # The trailing '_r' makes the colormap go dark-to-light instead of light-to-dark
#     cmap1 = plot.Colormap('violet red', name='pacific', fade=100, space='hsl')
#     # The color map has 256 colors.
#     colors = np.round(colors * 256)
#     return plot.to_rgb(cmap1(colors), space = 'hsl')

def get_force_vectors(linkage_pickle_name, omitBoundary = True):
    '''
    Compute the separation and tangential forces of a given linkage and return of list of force vector objects.
    '''
    # Create linkage object from pickle.
    linkage = pickle.load(gzip.open(linkage_pickle_name, 'r'))
    linkage.attraction_weight = 1e-5
    def eqm_callback(prob, i):
        pass
    elastic_rods.compute_equilibrium(linkage, callback = eqm_callback, options = OPTS)

    # Compute forces.
    AForceOnJoint = linkage.rivetNetForceAndTorques()[:, 0:3]
    if omitBoundary:
        for ji, j in enumerate(linkage.joints()):
            if (j.valence() < 4): AForceOnJoint[ji] = 0.0

    # Compute separation and tangential forces for each joint in the linkage. 
    separationForceVectors = []
    tangentialForceVectors = []
    epsilon = 1e-15
    for ji, j in enumerate(linkage.joints()):
        topSegmentIsA = 1 if j.type == j.Type.A_OVER_B else 0
        f = AForceOnJoint[ji]
        n = j.normal if topSegmentIsA else -j.normal # separation direction for this joint
        separation_force = np.clip(f.dot(n), 0, None) * n
        tangential_force = f - f.dot(n) * n
        if la.norm(separation_force) > epsilon:
            separationForceVectors.append(ForceVector(j.position, separation_force))
            separationForceVectors.append(ForceVector(j.position, -1 * separation_force))
        if la.norm(tangential_force) > epsilon:
            # Decomposed tangential force vector.
            # tf_A = np.dot(j.e_A, tangential_force) * j.e_A
            # tf_B = np.dot(j.e_B, tangential_force) * j.e_B
            # tangentialForceVectors.append(ForceVector(j.position, tf_A))
            # tangentialForceVectors.append(ForceVector(j.position, tf_B))
            tangentialForceVectors.append(ForceVector(j.position, tangential_force))
            tangentialForceVectors.append(ForceVector(j.position, -1 * tangential_force))
    return separationForceVectors, tangentialForceVectors

def write_force_vector_visualization_file(list_of_pickle, list_of_output_name):
    # Each element of this list is a pair of [separation_fv, tangential_fv].
    list_of_FV_pair = [get_force_vectors(name) for name in list_of_pickle]

    # Get Max Range.
    max_separation = np.amax(np.hstack([[fv.magnitude for fv in enum_fv[0]] for enum_fv in list_of_FV_pair]))
    max_tangential = np.amax(np.hstack([[fv.magnitude for fv in enum_fv[1]] for enum_fv in list_of_FV_pair]))
    print(max_separation, max_tangential)
    # Compute Color.
    def set_color(fv, norm):
        fv.color = get_color_scheme(fv.magnitude / norm)

    [[set_color(fv, max_separation) for fv in enum_fv[0]] for enum_fv in list_of_FV_pair]
    [[set_color(fv, max_tangential) for fv in enum_fv[1]] for enum_fv in list_of_FV_pair]

    # Normalize Force Magnitude to be Max 10.
    def normalize_magnitude(fv, norm):
        fv.magnitude = fv.magnitude / norm * 20

    [[normalize_magnitude(fv, max_separation) for fv in enum_fv[0]] for enum_fv in list_of_FV_pair]
    [[normalize_magnitude(fv, max_tangential) for fv in enum_fv[1]] for enum_fv in list_of_FV_pair]
  
    # Write To File.
    def write_to_file(name, fv_list):
        with open(name, 'w') as f:
            for fv in fv_list:
                f.write('{}, {}, {}, {}\n'.format(tuple(fv.direction), tuple(fv.joint_pos * 100), fv.magnitude, fv.color))

    [write_to_file('{}_separationForceVectors.txt'.format(list_of_output_name[i]), list_of_FV_pair[i][0]) for i in range(len(list_of_FV_pair))]
    [write_to_file('{}_tangentialForceVectors.txt'.format(list_of_output_name[i]), list_of_FV_pair[i][1]) for i in range(len(list_of_FV_pair))]