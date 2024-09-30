import numpy as np
import scipy
import sys, os
from numpy.linalg import norm as norm

import MeshFEM, triangulation, mesh, tri_mesh_viewer

def infer_surface(P, E, N, nonlinearIterations=0, nonlinearIterationViewer=None):
    """
    Infer a triangle surface patch filling the boundary curves specified by (P, E)
    and agreeing with the normal field specified by N.

    Parameters
    ----------
    P
        Boundary curve points
    E
        Boundary curve edges (indices into V)
    N
        Normals (evaluated at the boundary edge centers).

    Returns
    -------
    V, F
        Vertices, triangles of the interpolated surface patch triangle mesh.
    """
    # The plane fitting/registration operations below assume the center of mass is at the origin.
    center = np.mean(P, axis=0)
    P_centered = P - center

    P2d = project_onto_best_fit_plane(P_centered, np.mean(N, axis=0))

    # triangulate without inserting new points on the boundary (so we needn't interpolate normals)
    V, F, markers = triangulation.triangulate(P2d, E, triArea=0.1, flags="Y")
    V2d = V[:, 0:2]

    # Triangulation should keep the input points untouched *at the beginning of the vertex list*,
    # so we don't need to search for them in the input curve to lift them back to 3d.
    numInputPts = P2d.shape[0]
    assert((V2d[0:numInputPts] == P2d).all())

    m = mesh.Mesh(V, F)
    V3d = embed_boundary_tris(m, P_centered, N)
    V3d_linearized = linearized_plate_bending_minimizer(m, V3d)

    m.setVertices(V3d_linearized)

    V3d_nonlinear = V3d_linearized

    # Fine-tune with a nonlinear plate bending energy (needs a fully-built inflatables project!)
    V3d_nonlinear = nonlinear_bending_minimization(V2d, F, V3d_nonlinear, niter=nonlinearIterations, viewer=nonlinearIterationViewer)

    # Note: there was an additional rigid transformation applied when constructing the "metric fitter";
    # we undo this rigid transformation so the interpolated patch is positioned perfectly in space.
    from registration import register_points
    bdryIdxs = np.arange(0, numInputPts)
    R, t = register_points(P_centered[bdryIdxs], V3d_nonlinear[bdryIdxs])
    V3d_final =  (V3d_nonlinear @ R.T) + t + center[np.newaxis, :]
    V3d_final[bdryIdxs] = P[bdryIdxs]
    return (V3d_final, F)

def normalize(v):
    if (len(v.shape) == 2):
        n = norm(v, axis=1)
        return v / n[:, np.newaxis]
    l2 = norm(v)
    if (l2 < 1e-10): raise Exception('Zero vector')
    return v / l2

def project_onto_best_fit_plane(P, target_orientation):
    """
    Project points onto a plane of best fit.

    Parameters
    ----------
    P
        Points to project
    target_orientation
        The approximate desired normal of the plane; this is used to pick the
        *sign* of the plane of best fit.

    Returns
    -------
    P2D
        Best-fit points
    """
    # Get the eigenvectors of the covariance matrix;
    # The eigenvector for the smallest eigenvalue gives the plane's normal;
    # the other two eigenvectors form an orthonormal basis for the plane.
    eigenvectors = scipy.linalg.eigh(P.transpose() @ P)[1]

    # Pick the best-fit plane's orientation (normal sign) to agree with
    # Orient the best-fit plane to agree with the input normals.
    if (target_orientation.dot(eigenvectors[:, 0]) < 0): eigenvectors[:, 0] = -eigenvectors[:, 0]
    if np.linalg.det(eigenvectors) < 0: eigenvectors[:, [1, 2]] = eigenvectors[:, [2, 1]]
    tangents = eigenvectors[:, [1, 2]]

    # project points onto best-fit plane
    return P @ tangents

def embed_boundary_tris(m, P, N):
    """
    Compute an embedding of m's boundary-adjacent triangles that agrees with
    the boundary curves and their normal fields.
    For each boundary edge, conformally map the incident tri to its proper
    position, orientation in 3D according to the input edge vector and normal.

    Note: interior vertices appearing in multiple boundary-adjacent triangles (e.g., at the corners)
    will be repositioned multiple times; we arbitrarily keep the last position.

    Parameters
    ----------
    m
        Triangle mesh
    P
        Input boundary curve positions in 3D
    N
        Per-boundary-edge normal

    Returns
    -------
        V3d
            3D vertex positions for each mesh vertex, with boundary-adjacent
            triangle corners properly embedded. The other vertices' positions
            are unchanged.
    """
    eab = m.elementsAdjacentBoundary()

    V2d = m.vertices() # flat vertices
    V3d = V2d.copy()
    F = m.elements()
    N = normalize(N)

    for ti in eab:
        tri = F[ti]
        # cyclically permute triangle to put boundary edge at the beginning
        # (largest index at end)
        tri = np.roll(tri, 2 - np.argmax(tri))

        # Construct conformal map
        pts2d = V2d[tri]
        be2d = normalize(pts2d[1] - pts2d[0])
        n2d = np.array([0, 0, 1])
        R2d = np.column_stack([be2d, np.cross(n2d, be2d), n2d]) # rotation from canonical frame to 2D triangle's frame

        e = (tri[0], tri[1])
        p0 = P[e[0]]
        p1 = P[e[1]]
        be3d = normalize(p1 - p0)
        n3d = N[e[0]] # Edge indices coincide with the first endpoint index
        #assert(np.abs(n3d.dot(be3d)) < 1e-14)
        R3d = np.column_stack([be3d, np.cross(n3d, be3d), n3d]) # rotation from canonical frame to 3D triangle's frame

        R = R3d @ R2d.transpose()
        s = norm(p1 - p0) / norm(pts2d[1] - pts2d[0])
        mapTo3d = lambda p2d: p0 +  s * (R3d @ (R2d.transpose() @ (p2d - pts2d[0])))
        assert(norm(mapTo3d(pts2d[1]) - p1) < 1e-8)

        V3d[tri[0]] = p0
        V3d[tri[1]] = p1
        V3d[tri[2]] = mapTo3d(pts2d[2])
    return V3d

import differential_operators
def linearized_plate_bending_minimizer(m, V3d_initial):
    """
    Interpolate the boundary triangles positions within V3d_initial using a
    minimal Laplacian energy mesh.
    This Laplacian energy can be interpreted as a linearized thin plate bending
    energy which remains accurate as long as the surface does not stretch much.

    Parameters
    ----------
    m
        Triangle mesh
    V3d_initial
        Input vertex positions wherein the boundary-adjacent triangle corners
        are positioned as desired.

    Returns
    -------
    V3d
        Positions of all vertices that minimize the plate bending energy.
    """
    # Constrain the vertices of the boundary-adjacent triangles in the subsequent optimizations.
    constrainedVertices = np.unique(m.elements()[m.elementsAdjacentBoundary()].ravel())

    # Minimize the Laplacian energy with the boundary-adjacent triangles fixed.
    B = differential_operators.bilaplacian(m)

    mask = np.ones(m.numVertices(), dtype=np.bool)
    mask[constrainedVertices] = False

    B_ff = B[:,  mask][mask, :]
    B_fc = B[:, ~mask][mask, :]

    V3d = V3d_initial.copy()
    V3d[mask] = scipy.sparse.linalg.spsolve(B_ff, -B_fc @ V3d_initial[constrainedVertices])
    return V3d

def register(V3d_initial, V3d, bdryIndices):
    """
    Transform the interpolated surface patch back to the input location/orientation
    by registering its boundary points with the input points.

    Parameters
    ----------
    V3d_initial
        Points with which to align
    V3d
        Points to align
    bdryIndices
        Indices of boundary points

    Returns
    -------
    P2D
        Best-fit points
    """

def immersedMesh(fitter):
    return mesh.Mesh(fitter.getImmersion().transpose(), fitter.mesh().triangles())

class MetricFitVisMesh:
    def __init__(self, mfit):
        self.mfit = mfit
    def visualizationGeometry(self):
        return immersedMesh(self.mfit).visualizationGeometry()
    def visualizationField(self, data):
        return data

def nonlinear_bending_minimization(V2d, F, V3d_initial, niter=1, bendingStiffness = 10, viewer = None, newtonIterations=1000):
    """
    Minimize a nonlinear plate bending energy while penalizing a change in
    metric to void in-plane stretching.
    Repeatedly run this until triangles start collapsing into slivers.

    It probably would be better to use a conformal membrane energy term instead
    of metric fitting to allow the surface to stretch freely but prevent
    triangles from distorting. Currently, we simply let the surface
    incrementally stretch by resetting the target metric to the current metric
    on each call.

    Parameters
    ----------
    V2d, F
        Sheet triangle messh
    V3d_initial
        Initial immersion to fine-tune
    bendingStiffness
        Stiffness for the plate bending term. Increasing this will make a
        larger change to the shape in each iteration.
    viewer
        Optional 3D viewer to monitor progress

    Returns
    -------
        New positions for all mesh vertices that minimize the plate bending energy.
    """
    if (niter <= 0): return V3d_initial

    inflatables_dir = os.environ.get('INFLATABLES', '/Users/yren/Develop/EPFL_LGG/Inflatables')
    if (not os.path.isdir(inflatables_dir)): raise Exception('Inflatables not found')
    sys.path.append(f'{inflatables_dir}/python')
    import metric_fit
    from py_newton_optimizer import NewtonOptimizerOptions

    sheetMesh = metric_fit.Mesh(V2d, F)
    fitter = metric_fit.MetricFitter(sheetMesh)
    fitter.setImmersion(V3d_initial.transpose())
    fitter.setCurrentMetricAsTarget()

    fitter.bendingStiffness = bendingStiffness
    fitter.collapsePreventionWeight = 1.0

    constrainedVertices = np.unique(sheetMesh.elements()[sheetMesh.elementsAdjacentBoundary()].ravel())
    fixedVars = [var for v in constrainedVertices for var in 3*v + np.arange(3)]

    opt = NewtonOptimizerOptions()
    opt.gradTol = 1e-5
    opt.niter = newtonIterations
    fitter.bendingStiffness=10

    for it in range(niter):
        fitter.setCurrentMetricAsTarget(relativeCollapsePreventionThreshold=0.5)
        metric_fit.fit_metric_newton(fitter, fixedVars, opt)

        if (viewer is not None):
            import matplotlib
            mfVisMesh = MetricFitVisMesh(fitter)
            sf = tri_mesh_viewer.ScalarField(mfVisMesh, np.sqrt(fitter.metricDistSq()), colormap=matplotlib.cm.coolwarm, vmin=0)
            viewer.update(mesh=mfVisMesh, scalarField=sf)

    return fitter.getImmersion().transpose()
