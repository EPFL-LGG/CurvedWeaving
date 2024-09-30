"""
Compute integrated Gaussian curvature for a surface patch using several
different approaches:
    - Construct surface patch interpolant
        - Angle deficit formula
        - Cubic curvature
    - Boundary unflattening ("unbend Phi")
    - kb . d
    - 2 * atan(kb . d / 2)
    - d1 turning angle
"""
import numpy as np
import surface_inference
from surface_inference import normalize
import mesh, curvature
import vector_operations as vo

import igl
def quadricFittingCurvatures(mesh):
    d1, d2, kappa_1, kappa_2 = igl.principal_curvature(mesh.vertices(), mesh.triangles())
    return [kappa_1 * kappa_2, 0.5 * (kappa_1 + kappa_2)]

def curvatureComputations(P, E, N, D1, nonlinearIterations=0):
    result = {}
    V, F = surface_inference.infer_surface(P, E, N, nonlinearIterations=nonlinearIterations)
    m = mesh.Mesh(V, F)

    interior = np.ones(m.numVertices(), dtype=np.bool)
    interior[m.boundaryVertices()] = False

    gc = curvature.GaussianCurvatureSensitivity(m)
    result['Angle deficit'] = gc.integratedK()[interior].sum()

    K, H = quadricFittingCurvatures(m)
    # Exact integral of piecewise linear K field over the mesh
    result['Quadric fitting'] = K[m.elements()].mean(axis=1).dot(m.elementVolumes())

    T = normalize(P[E[:, 1]] - P[E[:, 0]])
    N = normalize(N)
    D1 = normalize(D1)

    assert(np.max(np.abs((T @ N.T).diagonal())) < 1e-14)

    numEdges = P.shape[0]
    angles  = np.array([vo.signedAngle(T[i - 1], vo.parallelTransportNormalized(N[i], N[i - 1], T[i]),
                                       N[i - 1]) for i in range(numEdges)])
    angleKb = np.array([vo.curvatureBinormal(T[i - 1], T[i]).dot(normalize(N[i - 1] + N[i])) for i in range(numEdges)])
    anglePhi = 2 * np.arctan(0.5 * angleKb)

    def signedAngleNonorthogonal(v1, v2, n): # Doesn't assume n is parallel to v1.cross(V2)
        # Compute signed angle after projecting onto plane with normal n
        v1 = v1 - n.dot(v1) * n
        v2 = v2 - n.dot(v2) * n
        return vo.angle(v1, v2) * np.sign(np.cross(v1, v2).dot(n))

    angleD1  = np.array([signedAngleNonorthogonal(D1[i - 1], D1[i], normalize(N[i - 1] + N[i])) for i in range(numEdges)])

    result['unbendPhi']             = 2 * np.pi - angles.sum()
    result['Kb . d2']               = 2 * np.pi - angleKb.sum()
    result['2 * atan(Kb . d2 / 2)'] = 2 * np.pi - anglePhi.sum()
    result['d1 angle']              = 2 * np.pi - angleD1.sum()

    return result
