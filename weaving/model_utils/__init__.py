#########################
#  auto-generated files #
#########################

import sys as _sys
_sys.path.insert(0, '/Users/yren/Develop/EPFL_LGG/elastic_rods/3rdparty/MeshFEM/python/')

# Content from template file /Users/yren/Develop/EPFL_LGG/elastic_rods/3rdparty/MeshFEM/python/py_module_init.template #

import sparse_matrices
import energy
import mesh

from mesh_building import *
from energy_building import *
import tri_mesh_viewer

# may not have elastic_structure build if Boost is not presented
try:
    import elastic_structure
    from elastic_structure_building import *
except:
    pass