elastic_rods_dir = "../../elastic_rods/python"
weaving_dir = "../"
import os.path as osp
import sys

sys.path.append(elastic_rods_dir)
sys.path.append(weaving_dir)
import numpy as np, elastic_rods, linkage_vis
import numpy.linalg as la
import os
import networkx as nx
from matplotlib import pyplot as plt


default_camera_parameters = (
    (3.466009282140468, -4.674139805388271, -2.556131049738206),
    (-0.21402574298422497, -0.06407538766530313, -0.9747681088523519),
    (0.1111, 0.1865, 0.5316),
)
RIBBON_CS = [1, 10]
ISO_CS = [4, 4]
MODEL_PATH = osp.join(weaving_dir + "models/equal_edge_atI.obj")
SUBDIVISION_RESOLUTION = 5


def initialize_linkage(cross_section=ISO_CS,
                       subdivision_res=SUBDIVISION_RESOLUTION,
                       model_path=MODEL_PATH,
                       cam_param=default_camera_parameters):
    """[summary]

    Parameters
    ----------
    cross_section : [type], optional
        [description], by default ISO_CS
    subdivision_res : [type], optional
        [description], by default SUBDIVISION_RESOLUTION
    model_path : [type], optional
        [description], by default MODEL_PATH
    cam_param : [type], optional
        [description], by default default_camera_parameters

    Returns
    -------
    [type]
        [description]
    """
    l = elastic_rods.RodLinkage(model_path, subdivision_res, False)
    driver = l.centralJoint()
    l.setMaterial(
        elastic_rods.RodMaterial(
            "rectangle", 2000, 0.3, cross_section, stiffAxis=elastic_rods.StiffAxis.D1
        )
    )
    l.set_design_parameter_config(use_restLen=True, use_restKappa=False)
    elastic_rods.designParameter_solve(l, regularization_weight=0.1)
    jdo = l.dofOffsetForJoint(driver)
    fixedVars = list(range(jdo, jdo + 6))  # fix rigid motion for a single joint
    elastic_rods.compute_equilibrium(l, fixedVars=fixedVars)
    view = linkage_vis.LinkageViewer(l, width=1024, height=640)
    view.setCameraParams(cam_param)
    return l, view


def update_rest_curvature(linkage):
    """[summary]

    Parameters
    ----------
    linkage : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """
    # Always use ISO_CS in update rest curvature because the fixed point iteration want to extract the geodesic curvature occured.
    linkage.setMaterial(
        elastic_rods.RodMaterial(
            "rectangle", 2000, 0.3, ISO_CS, stiffAxis=elastic_rods.StiffAxis.D1
        )
    )
    elastic_rods.designParameter_solve(linkage, regularization_weight=0.1)
    #         driver =linkage.centralJoint()
    #         jdo = linkage.dofOffsetForJoint(driver)
    #         fixedVars = list(range(jdo, jdo + 6)) # fix rigid motion for a single joint
    elastic_rods.compute_equilibrium(
        linkage, fixedVars=linkage.jointPositionDoFIndices()
    )
    for i in range(linkage.numSegments()):
        r = linkage.segment(i).rod
        # compute new kappa
        curr_rest_kappa = r.restKappas()
        deformed_kappa = r.deformedConfiguration().kappa
        new_kappa = []
        for j in range(len(curr_rest_kappa)):
            new_kappa.append(np.array([deformed_kappa[j][0], curr_rest_kappa[j][1]]))

        linkage.segment(i).rod.setRestKappas(new_kappa)

    return linkage


def set_ribbon_linkage(linkage, cam_param=default_camera_parameters):
    """[summary]

    Parameters
    ----------
    linkage : [type]
        [description]
    cam_param : [type], optional
        [description], by default default_camera_parameters

    Returns
    -------
    [type]
        [description]
    """
    linkage.setMaterial(
        elastic_rods.RodMaterial(
            "rectangle", 2000, 0.3, RIBBON_CS, stiffAxis=elastic_rods.StiffAxis.D1
        )
    )
    driver = linkage.centralJoint()
    elastic_rods.designParameter_solve(linkage, regularization_weight=0.1)
    jdo = linkage.dofOffsetForJoint(driver)
    fixedVars = list(range(jdo, jdo + 6))  # fix rigid motion for a single joint
    elastic_rods.compute_equilibrium(linkage, fixedVars=fixedVars)
    view = linkage_vis.LinkageViewer(linkage, width=1024, height=640)
    view.setCameraParams(cam_param)
    return linkage, view

def export_linkage_geometry_to_obj(linkage, filename, vd = [], use_color = False, colors = [], scale = 1):
    """Export a linkage to a .obj file

    Parameters
    ----------
    linkage : linkage object
        linkage object to export.
    filename : string
        name of the exported file. Extension will be automatically added.
    vd : list, optional
        [description], by default []
    use_color : bool, optional
        [description], by default False
    colors : list, optional
        [description], by default []
    """
    geometry = linkage.visualizationGeometry()
    if filename[-4:] != ".obj":
        filename = "{}.obj".format(filename)
    with open(filename, "w") as f:
        points = geometry[0]
        faces = geometry[1]
        normals = geometry[2]
        for idx, point in enumerate(points):
            if not use_color:
                f.write('v {} {} {}\n'.format(
                    point[0] * scale, 
                    point[1] * scale, 
                    point[2] * scale))
            else:
                f.write('v {} {} {} {} {} {}\n'.format(
                    point[0] * scale, 
                    point[1] * scale, 
                    point[2] * scale, 
                    colors[idx][0], 
                    colors[idx][1], 
                    colors[idx][2]))
        for normal in normals:
            f.write("vn {} {} {}\n".format(normal[0], normal[1], normal[2]))
        for face in faces:
            f.write(
                "f {} {} {}\n".format(
                    int(face[0]) + 1, int(face[1]) + 1, int(face[2]) + 1
                )
            )
        for dis in vd:
            f.write("vd {}\n".format(dis))


def get_ribbon_family(linkage, ribbons):
    """Build graph for a ribbon family

    Parameters
    ----------
    linkage : [type]
        [description]
    ribbons : [type]
        [description]

    Returns
    -------
    [type]
        [description]
    """

    segment_in_ribbon = -1 * np.ones(linkage.numSegments())
    for ri in range(len(ribbons)):
        for (seg_index, _) in ribbons[ri]:
            segment_in_ribbon[seg_index] = ri

    G = nx.Graph()
    G.add_nodes_from(range(len(ribbons)))
    for ji in range(linkage.numJoints()):
        seg_1 = linkage.joint(ji).segments_A[0]
        seg_2 = linkage.joint(ji).segments_B[0]
        ribbon_1 = int(segment_in_ribbon[seg_1])
        ribbon_2 = int(segment_in_ribbon[seg_2])
        G.add_edge(ribbon_1, ribbon_2)

    ribbon_family = list(-1 * np.ones(len(ribbons)))
    ribbon_family[0] = "A"
    neighbor = [n for n in G[0]][0]
    ribbon_family[neighbor] = "B"

    C_family = sorted(nx.common_neighbors(G, 0, neighbor))
    for index in C_family:
        ribbon_family[index] = "C"

    B_family = sorted(nx.common_neighbors(G, 0, C_family[0]))
    for index in B_family:
        ribbon_family[index] = "B"

    A_family = sorted(nx.common_neighbors(G, B_family[0], C_family[0]))
    for index in A_family:
        ribbon_family[index] = "A"

    for a_e in A_family:
        for b_e in B_family:
            C_family.extend(sorted(nx.common_neighbors(G, a_e, b_e)))
            C_family = list(set(C_family))
    for index in C_family:
        ribbon_family[index] = "C"

    for a_e in A_family:
        for c_e in C_family:
            B_family.extend(sorted(nx.common_neighbors(G, a_e, c_e)))
            B_family = list(set(B_family))
    for index in B_family:
        ribbon_family[index] = "B"

    for b_e in B_family:
        for c_e in C_family:
            A_family.extend(sorted(nx.common_neighbors(G, b_e, c_e)))
            A_family = list(set(A_family))
    for index in A_family:
        ribbon_family[index] = "A"
    assert len(A_family) + len(B_family) + len(C_family) == len(ribbons)
    return ribbon_family

# FIXME: TO DITCH OR RENAME FOR PUBLIC RELEASE
def write_linkage_ribbon_output_florin(linkage, ribbons, resolution, RIBBON_NAME,
                                       use_family_label=False,
                                       scale=1,
                                       distance_scale=1, 
                                       write_stress = False):
    """[summary]

    Parameters
    ----------
    linkage : [type]
        [description]
    ribbons : [type]
        [description]
    resolution : [type]
        [description]
    RIBBON_NAME : [type]
        [description]
    use_family_label : bool, optional
        [description], by default False
    scale : int, optional
        [description], by default 1
    distance_scale : int, optional
        [description], by default 1

    Returns
    -------
    [type]
        [description]
    """

    ribbon_centerline_index = []
    ribbon_centerline_points = []
    ribbon_centerline_normal = []
    ribbon_centerline_cross_section = []
    ribbon_bending_stress = []
    ribbon_twisting_stress = []

    ribbon_loop_indicator = []
    curr_offset = 0

    ribbon_family = range(len(ribbons))
    if use_family_label:
        ribbon_family = get_ribbon_family(linkage, ribbons)

    if write_stress:
        twisting_stress = linkage.twistingStresses()
        bending_stress = linkage.maxBendingStresses()
    def normalize(vec):
        return vec / la.norm(vec)

    def is_valid_segment_index(index):
        return index < linkage.numSegments()

    def is_valid_joint_index(index):
        return index < linkage.numJoints()

    def get_cross_section(rod, index):
        m = rod.material(index)
        [a1, a2, a3, a4] = m.crossSectionBoundaryPts
        width = la.norm(a1 - a4)
        thickness = la.norm(a1 - a2)
        return np.array([width, thickness])

    for ribbon in ribbons:
        curr_ribbon_centerline_index = []

        curr_start_joint = linkage.segment(ribbon[0][0]).startJoint
        if curr_start_joint < linkage.numJoints():
            prev_segment = linkage.joint(curr_start_joint).continuationSegment(ribbon[0][0])
            if is_valid_segment_index(prev_segment):
                ribbon_loop_indicator.append("L")
            else:
                ribbon_loop_indicator.append("O")
        else:
            ribbon_loop_indicator.append("O")

        for (segment_index, orientation) in ribbon:
            if orientation != 1:
                print("Rod orientation incorrect!")
            curr_rod = linkage.segment(segment_index).rod
            dc = curr_rod.deformedConfiguration()
            curr_start_joint = linkage.segment(segment_index).startJoint
            curr_end_joint = linkage.segment(segment_index).endJoint
            if write_stress:
                curr_rod_bending_stress = bending_stress[segment_index]
                curr_rod_twisting_stress = twisting_stress[segment_index]

            is_open_end = False
            if not is_valid_joint_index(curr_start_joint):
                is_open_end = True
            else:
                prev_segment = linkage.joint(curr_start_joint).continuationSegment(segment_index)
                if not is_valid_segment_index(prev_segment):
                    is_open_end = True

            count_point = 0
            if is_open_end:
                ribbon_centerline_points.append(curr_rod.deformedPoints()[0])
                ribbon_centerline_normal.append(dc.materialFrame[0].d2)
                ribbon_centerline_cross_section.append(get_cross_section(curr_rod, 0))
                if write_stress:
                    ribbon_bending_stress.append(curr_rod_bending_stress[0])
                    ribbon_twisting_stress.append(curr_rod_twisting_stress[0])
                count_point += 1

            ribbon_centerline_points.extend(curr_rod.deformedPoints()[1:-1])
            ribbon_centerline_normal.extend(
                [normalize(dc.materialFrame[edge_index - 1].d2 + dc.materialFrame[edge_index].d2)
                    for edge_index in range(resolution)[1:]]
            )
            ribbon_centerline_cross_section.extend(
                [(get_cross_section(curr_rod, edge_index - 1) + get_cross_section(curr_rod, edge_index)) / 2
                    for edge_index in range(resolution)[1:]]
            )
            if write_stress:
                ribbon_bending_stress.extend(curr_rod_bending_stress[1:-1])
                ribbon_twisting_stress.extend(curr_rod_twisting_stress[1:-1])
            count_point += resolution - 1

            is_end_of_rod = False
            if not is_valid_joint_index(curr_end_joint):
                is_end_of_rod = True
            else:
                next_segment = linkage.joint(curr_end_joint).continuationSegment(segment_index)
                if not is_valid_segment_index(next_segment):
                    is_end_of_rod = True
                    
            if is_end_of_rod:
                ribbon_centerline_points.append(curr_rod.deformedPoints()[-1])
                ribbon_centerline_normal.append(dc.materialFrame[-1].d2)
                ribbon_centerline_cross_section.append(get_cross_section(curr_rod, curr_rod.numEdges() - 1))
                if write_stress:
                    ribbon_bending_stress.append(curr_rod_bending_stress[-1])
                    ribbon_twisting_stress.append(curr_rod_twisting_stress[-1])
                count_point += 1

            curr_centerline_index = np.arange(curr_offset, curr_offset + count_point)
            curr_offset += count_point

            curr_ribbon_centerline_index.extend(curr_centerline_index)
        ribbon_centerline_index.append(curr_ribbon_centerline_index)

    if not os.path.exists(RIBBON_NAME):
        os.makedirs(RIBBON_NAME)

    with open("{}/{}_polylines.txt".format(RIBBON_NAME, RIBBON_NAME), "w") as f:
        ribbon_count = 0
        for line in ribbon_centerline_index:
            f.write(
                "{} {} {}\n".format(
                    ribbon_family[ribbon_count],
                    ribbon_loop_indicator[ribbon_count],
                    " ".join([str(x) for x in line]),
                )
            )
            ribbon_count += 1

    with open("{}/{}_points.txt".format(RIBBON_NAME, RIBBON_NAME), "w") as f:
        for point in ribbon_centerline_points:
            f.write("{}\n".format(" ".join([str(x * scale) for x in list(point)])))

    with open("{}/{}_normals.txt".format(RIBBON_NAME, RIBBON_NAME), "w") as f:
        for normal in ribbon_centerline_normal:
            f.write("{}\n".format(" ".join([str(x) for x in list(normal)])))

    with open("{}/{}_cross_section.txt".format(RIBBON_NAME, RIBBON_NAME), "w") as f:
        for cross_section in ribbon_centerline_cross_section:
            f.write(
                "{}\n".format(" ".join([str(x * scale) for x in list(cross_section)]))
            )
    with open('{}/{}_centerline.obj'.format(RIBBON_NAME, RIBBON_NAME), 'w') as f:
        for point in ribbon_centerline_points:
            f.write('v {}\n'.format(' '.join([str(x) for x in list(point)])))

        for line in ribbon_centerline_index:
            for i in range(len(line) - 1):
                f.write('l {} {}\n'.format(line[i]+1, line[i+1]+1))

    print(len(ribbon_centerline_points), len(ribbon_bending_stress))

    if write_stress:
        write_stress_texture(ribbon_bending_stress, ribbon_centerline_index, RIBBON_NAME)
        write_stress_texture(ribbon_twisting_stress, ribbon_centerline_index, RIBBON_NAME)

    return ribbon_centerline_index, ribbon_centerline_points, ribbon_centerline_normal, ribbon_centerline_cross_section, ribbon_bending_stress, ribbon_twisting_stress


def write_stress_texture(stresses, allPolylines, ribbon_name, stressMin = None, stressMax = None):
    print('write stress')
    # write stress images
    resolution = 1000
    
    if (stressMin is None): stressMin = np.amin(np.hstack((stresses)));
    if (stressMax is None): stressMax = np.amax(np.hstack((stresses)));
    normalize = plt.Normalize(vmin=stressMin, vmax=stressMax)
    cmap = plt.cm.plasma

    for idx, polyline in enumerate(allPolylines):
        nv = len(polyline)
        interpolated_stresses = np.interp(np.linspace(0, nv - 1, resolution), range(nv), [stresses[vi] for vi in polyline])
        # interpolated_stresses = np.linspace(0, stressMax, resolution)
        # interpolated_stresses[0:10] = stressMax / 2
        # interpolated_stresses[resolution - 10:] = stressMax / 2
        image = cmap(normalize(np.array(interpolated_stresses).reshape(1, resolution)))
        plt.imsave('{}/{}_stress_{}.png'.format(ribbon_name, ribbon_name, idx), image)

def write_distance_to_linkage_mesh(linkage, width, model_name, return_distance_field = False):
    def get_color_scheme(colors):
        cmap = plt.cm.plasma
    #     cmap = cmap2
        return cmap(colors)
    distance_to_surface = np.array(linkage.get_squared_distance_to_target_surface((linkage.visualizationGeometry()[0]).flatten()))
    distance_to_surface = np.sqrt(distance_to_surface)
    ribbon_distance_scale = 1/width
    # print(ribbon_distance_scale)
    distance_to_surface *= ribbon_distance_scale
    if return_distance_field:
        return get_color_scheme(distance_to_surface)
    export_linkage_geometry_to_obj(linkage, 'distance_mesh_{}.obj'.format(model_name), vd = distance_to_surface, use_color=True, colors = get_color_scheme(distance_to_surface))

    # distance_to_surface = linkage.get_squared_distance_to_target_surface(np.array(ribbon_centerline_points).flatten())
    # distance_to_surface = np.sqrt(distance_to_surface)
    # cmap = plt.cm.viridis

    # texture_res = 1000
    # for idx, polyline in enumerate(ribbon_centerline_index):
    #     nv = len(polyline)
    #     interpolated_distance = np.interp(np.linspace(0, nv - 1, texture_res), range(nv), [distance_to_surface[vi] for vi in polyline])
    #     # interpolated_distance = np.linspace(0, stressMax, resolution)
    #     # interpolated_distance[0:10] = stressMax / 2
    #     # interpolated_distance[resolution - 10:] = stressMax / 2
    #     image = cmap(np.array(interpolated_distance).reshape(1, texture_res)/distance_scale)
    #     print(np.max((np.array(interpolated_distance).reshape(1, texture_res))/distance_scale))
    #     plt.imsave('{}/{}_distance_{}.png'.format(RIBBON_NAME, RIBBON_NAME, idx), image)


def write_centerline_normal_deviation_to_linkage_mesh(linkage, model_name):
    print('write normal deviation mesh')
    def get_color_scheme(colors):
        cmap = plt.cm.plasma
    #     cmap = cmap2
        return cmap(colors)
    linkage_centerline_normals = np.array(linkage.visualizationGeometry()[2])
    linkage_centerline_projection_normals = np.array(linkage.get_closest_point_normal((linkage.visualizationGeometry()[0]).flatten()))
    linkage_centerline_projection_normals = linkage_centerline_projection_normals.reshape(linkage_centerline_normals.shape)
    deviation_angle = []
    for i in range(len(linkage_centerline_normals)):
        closeness = np.dot(linkage_centerline_projection_normals[i], linkage_centerline_normals[i])
        # The value is between 0 and pi/2; scale to between 0 and 1
        angle = min(np.arccos(abs(closeness))/ (np.pi/2), 1)
        if (1-angle) < 0.4:
            angle = 0
        if (angle > 1 or angle < 0):
            print("Wrong normal deviation angle!")
        deviation_angle.append(angle)
    print("max deviation angle: ", max(deviation_angle))
    deviation_angle = np.array(deviation_angle) / max(deviation_angle)
    export_linkage_geometry_to_obj(linkage, 'normal_deviation_mesh_{}.obj'.format(model_name), vd = deviation_angle, use_color=True, colors = get_color_scheme(deviation_angle))
    return deviation_angle












