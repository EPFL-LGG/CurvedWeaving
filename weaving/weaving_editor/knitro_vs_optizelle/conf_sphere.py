import os
import os.path as osp

elastic_rods_dir = '../../../elastic_rods/python/'
weaving_dir = '../../'

MODEL_NAME = "sphere_1"
MODEL_PATH = osp.join(weaving_dir + 'normalized_objs/models/{}.obj'.format(MODEL_NAME))
SUBDIVISION_RESOLUTION = 20
INPUT_SURFACE_PATH = osp.join(weaving_dir + 'normalized_objs/surface_models/{}.obj'.format(MODEL_NAME))

rod_length = 0.3534025445286393
width = rod_length / 15 * 3
thickness = width / 5 * 0.35
RIBBON_CS = [thickness, width]
minRestLen = RIBBON_CS[1] * 1.0

# Regularization parameters

rw = 0.1
sw = 0.1

# Optizelle params specific to each cases

optzel_trust_radius = 1.0
# Interior point barrier height
optzel_mu = 0.6
