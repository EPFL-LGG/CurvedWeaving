import numpy as np
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/31735499/calculate-angle-clockwise-between-two-points
from math import acos
from math import sqrt
from math import pi

elastic_rods_dir = '../elastic_rods/python'
import sys; sys.path.append(elastic_rods_dir)
from linkage_utils import order_segments_by_ribbons, get_turning_angle_and_length_from_ordered_rods
import numpy.linalg as la
import numpy as np, elastic_rods, linkage_vis
from elastic_rods import EnergyType, InterleavingType
import os

def length(v):
    return sqrt(v[0]**2+v[1]**2)
def dot_product(v,w):
    return v[0]*w[0]+v[1]*w[1]
def determinant(v,w):
    return v[0]*w[1]-v[1]*w[0]
def inner_angle(v,w):
    cosx=dot_product(v,w)/(length(v)*length(w))
    rad=acos(cosx) # in radians
    return rad*180/pi # returns degrees
def angle_clockwise(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner / 180 * np.pi
    else: # if the det > 0 then A is immediately clockwise of B
        return (360-inner) / 180 * np.pi
#     ----------------

def angle_counterclockwise(A, B):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det>0: #this is a property of the det. If the det > 0 then B is counterclockwise of A
        return inner / 180 * np.pi
    else: # if the det > 0 then A is immediately clockwise of B
        return (360-inner) / 180 * np.pi
#     ------------
def match_geo_curvature_and_edge_len(linkage):
    all_ribbon_curvatures = []
    all_ribbon_edge_len = []
    n_strip = 10
    n_segment_per_strip = int(linkage.numSegments() / n_strip)
    for i in range(n_strip):
        geo_curvatures = []
        edge_len = []
        for seg_index in range(i * n_segment_per_strip, (i+1) * n_segment_per_strip):
            curr_curvature = linkage.segment(seg_index).rod.deformedConfiguration().kappa
            geo_curvatures.extend(np.array(curr_curvature)[:, 0])
            
            curr_edge_lens = linkage.segment(seg_index).rod.deformedConfiguration().len
            matched_edge_lens = [curr_edge_lens[0] / 2]
            for edge_index in range(len(curr_edge_lens) - 1):
                matched_edge_lens.append((curr_edge_lens[edge_index] + curr_edge_lens[edge_index+1])/2)
            matched_edge_lens.append(curr_edge_lens[-1]/2)
            edge_len.extend(matched_edge_lens)
        print(len(geo_curvatures), len(edge_len))
        all_ribbon_curvatures.append(geo_curvatures)
        all_ribbon_edge_len.append(edge_len)
    return all_ribbon_curvatures, all_ribbon_edge_len

def rotate(origin, point, angle):
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return np.array([qx, qy])

def get_curve_from_angle(angles, edge_lens, widths, segment_res, with_extra_tip = False):
    if with_extra_tip:
        angles = angles[-int(segment_res / 2) :] + angles + angles[:int(segment_res / 2)]
        edge_lens = edge_lens[-int(segment_res / 2) :] + edge_lens + edge_lens[:int(segment_res / 2)]
        widths = widths[-int(segment_res / 2) :] + widths + widths[:int(segment_res / 2)]
    curr_end_point = np.array([0, 0])
    curr_angle = 0
    curve_point_list = []
    for i in range(len(angles)):
        curr_angle += angles[i]
        curr_angle %= 2 * np.pi
        edge = np.array([edge_lens[i], 0]) + curr_end_point
        curr_end_point = rotate(curr_end_point, edge, curr_angle)
        curve_point_list.append(curr_end_point)
    curve_point_list = np.array(curve_point_list)
    # min_index = np.array(angles).argmin()
    # second_min_index = np.argpartition(np.array(angles), 1)[1]
    rotated_vector = curve_point_list[-1] - curve_point_list[0]
    # Module by pi so that the curve's x coordinates goes from small to large
    rotated_angle = angle_clockwise(rotated_vector, [1, 0]) % np.pi
    curve_point_list = np.array([rotate([0, 0], point, -rotated_angle) for point in curve_point_list])
    xs, ys = curve_point_list[:, 0], curve_point_list[:, 1]
    # plt.clf()
    # plt.axis('equal')
    # plt.plot(xs, ys)
    # plt.savefig('flatten_shape.png', format='png', dpi = 320)
    # plt.show()

    return xs, ys, widths

def get_laser_cutting_pattern_data(ribbons, all_ribbon_angle, all_ribbon_edge_len, all_ribbon_num_seg, scale, width, all_ribbon_widths, resolution, ribbon_index = 0):
    rod_angle, rod_length, num_seg_per_rod, rod_width = all_ribbon_angle[ribbon_index], all_ribbon_edge_len[ribbon_index], all_ribbon_num_seg[ribbon_index], all_ribbon_widths[ribbon_index]
    xs, ys, widths = get_curve_from_angle(rod_angle, rod_length, rod_width, resolution, True)
    # print("ribbon centerline: ", xs, ys)
    # print("scale: ", scale)
    print("num seg per rod", num_seg_per_rod)
    # print("num ribbons", len(ribbons))
    # print("rod length size: ", len(rod_length))
    # print("rod length: ", [sum(rod_length[i * (resolution - 1) : (i + 1) * (resolution - 1)]) for i in range(num_seg_per_rod-1)])
    # print(min([sum(rod_length[i * (resolution - 1) : (i + 1) * (resolution - 1)]) for i in range(num_seg_per_rod-1)]) * scale)
    
    # Assume the rod are uniformly sampled.
    point_per_segment = int(len(all_ribbon_edge_len[ribbon_index]) / num_seg_per_rod)

    def scale_to_frame(xs, ys):
        scaled_xs = np.array([(x - min(xs)) * scale for x in xs])
        scaled_ys = np.array([y * scale for y in ys])
        return scaled_xs, scaled_ys
    
    def get_point(index):
        return np.array([scaled_xs[index], scaled_ys[index]])
    
    def get_extension(index):
        return extend_direction_list[index] * widths[index]

    def rotate_90_ccw(vec):
        x, y = vec[0], vec[1]
        angle = np.pi / 2
        n_x = x * np.cos(angle) + y * np.sin(angle)
        n_y = -x * np.sin(angle) + y * np.cos(angle)
        return np.array([n_x, n_y])
    
    scaled_xs, scaled_ys = scale_to_frame(xs, ys)
    extend_direction_list = []
    for i in range(len(scaled_ys))[1:-1]:
        prev_point, next_point, curr_point = get_point(i-1), get_point(i+1), get_point(i)
        to_prev, to_next = prev_point - curr_point, next_point - curr_point
        to_prev /= la.norm(to_prev)
        to_next /= la.norm(to_next)
        extend_direction = to_prev + to_next
        if la.norm(extend_direction) <= 1e-7:
            extend_direction = rotate_90_ccw(to_next)
        extend_direction /= la.norm(extend_direction)
        correct_sign = np.sign(np.cross(to_next, extend_direction))
        if correct_sign == 0:
            print("Warning: the centerline is exactly straight.")
        extend_direction *= correct_sign
        extend_direction_list.append(extend_direction)
    extend_direction_list = extend_direction_list[:1] + extend_direction_list + extend_direction_list[-1:]
    upper_point_list, lower_point_list = [], []
    for i in range(len(extend_direction_list)):
        curr_point = get_point(i)      
        upper_point = curr_point + get_extension(i)
        lower_point = curr_point - get_extension(i)
        upper_point_list.append(upper_point)
        lower_point_list.append(lower_point)
    upper_point_list, lower_point_list = np.array(upper_point_list), np.array(lower_point_list)

#     The joint position since the joint is in the middle of the segments
#     Assume the rod segment are concatenated together to form the longer rod, the joint positions are at the start of each segment. 
    tip_size = int(resolution / 2)
    joints = np.array([(get_point(i * point_per_segment + tip_size - 1) + get_point(i * point_per_segment - 2 + tip_size)) / 2 for i in range(num_seg_per_rod + 1)])
    upper_joints = np.array([(get_point(i * point_per_segment + tip_size - 1) + get_point(i * point_per_segment - 2 + tip_size) + get_extension(i * point_per_segment + tip_size - 1) + get_extension(i * point_per_segment - 2 + tip_size)) / 2  for i in range(num_seg_per_rod + 1)])
    lower_joints = np.array([(get_point(i * point_per_segment + tip_size - 1) + get_point(i * point_per_segment - 2 + tip_size) - get_extension(i * point_per_segment + tip_size - 1) - get_extension(i * point_per_segment - 2 + tip_size)) / 2  for i in range(num_seg_per_rod + 1)])


    joints_angle = [angle_clockwise(np.array([1, 0]), get_point(i * point_per_segment + tip_size - 1) - get_point(i * point_per_segment - 2 + tip_size)) for i in range(num_seg_per_rod + 1)] 
    joints_direction = [get_point(i * point_per_segment + tip_size - 1 + int(tip_size / 4)) - get_point(i * point_per_segment - 1 - int(tip_size / 4) + tip_size) for i in range(num_seg_per_rod + 1)] 
    joints_direction = [d / la.norm(d) for d in joints_direction]

    plt.clf()
    f = plt.figure(1)
    ax = f.add_subplot(1,1,1)
    plt.gcf().subplots_adjust(0,0,1,1)
    ax.set_aspect('equal')
    ax.axis('off')
    top_x = upper_point_list[:, 0]
    top_y = upper_point_list[:, 1]
    bottom_x = lower_point_list[:, 0]
    bottom_y = lower_point_list[:, 1]
    joints_xs, joints_ys = joints[:, 0], joints[:, 1]
    upper_joints_xs, upper_joints_ys = upper_joints[:, 0], upper_joints[:, 1]
    lower_joints_xs, lower_joints_ys = lower_joints[:, 0], lower_joints[:, 1]

#     np.savez('ground_truth_sphere_laser_cutting_pattern.npz', top_x = np.array(top_x), top_y = np.array(top_y), bottom_x = np.array(bottom_x), bottom_y = np.array(bottom_y), joints_xs = np.array(joints_xs), joints_ys = np.array(joints_ys))
    return top_x, top_y, bottom_x, bottom_y, joints_xs, joints_ys, joints_angle, joints_direction, upper_joints_xs, upper_joints_ys, lower_joints_xs, lower_joints_ys

def get_slit_orientation(linkage, surface_path, model_path, subdivision_resolution):
    ribbons = order_segments_by_ribbons(linkage)
    all_joint_index = [[linkage.segment(curr_ribbon[i][0]).startJoint for i in range(len(curr_ribbon))] for curr_ribbon in ribbons]
    offset_linkage = elastic_rods.SurfaceAttractedLinkage(surface_path, True, model_path, subdivision_resolution, False, InterleavingType.weaving)
    slit_orientation = []
    for ri in range(len(ribbons)):
        curr_slit_ori = []
        for ji in range(len(all_joint_index[ri])):
            curr_joint = offset_linkage.joint(all_joint_index[ri][ji])
            if ((curr_joint.type == 1) and (ribbons[ri][ji][0] in curr_joint.segments_A)) or ((curr_joint.type == -1) and (ribbons[ri][ji][0] in curr_joint.segments_B)):
                curr_slit_ori.append(1)
            else:
                curr_slit_ori.append(-1)
        slit_orientation.append(curr_slit_ori + [curr_slit_ori[0]])
    return slit_orientation

def get_all_curve_svg(file_name, top_x_list, top_y_list, bottom_x_list, bottom_y_list, joint_x_list, joint_y_list, joint_index_list_list, ribbon_index_list, joint_angle_list_list, all_joint_direction_list, upper_joint_x_list, upper_joint_y_list, lower_joint_x_list, lower_joint_y_list, thickness, use_slit, slit_orientation):
    rHole  = 1.1/2                           # radius of holes at intersection
    srHole = 0.25                           # radius of intermediate holes
# Parameters related to laser cutting file generation
    Offset = 10.              # Move strip 1 from 0,0 (since it will be off the page)
    stroke = 0.005              # stroke width (0.005 for actual file, larger number for debugging)
    color_line = 'red'   # line color
    color_hole = 'red'   # hole color, it is advicable to cut the holes first because once the ribbons are cut, they may fly away due to the fan
    pre='<path d="'
    post='" stroke="' + color_line +'" stroke-width="'+str ( stroke)+'" fill-opacity="0" />'
    pm = 2.83465 # convert from point to mm

    y_offset = 0
    max_x = 0
    
    # Gather the data about the upper and lower boundary of the ribbons. We need the information about the offset for marking and writing the crossings, but we want to cut these lines last. 
    svgTList=[]
    for list_index in range(len(top_x_list)):
        top_x, top_y, bottom_x, bottom_y, joint_x, joint_y, joint_index_list, ribbon_index= top_x_list[list_index], top_y_list[list_index], bottom_x_list[list_index], bottom_y_list[list_index], joint_x_list[list_index], joint_y_list[list_index], joint_index_list_list[list_index], ribbon_index_list[list_index]
        
        y_offset += 0 - min(top_y) + 10
        svgT = ['M'+str(top_x[0]*pm + Offset)+','+str(top_y[0]*pm + Offset + y_offset * pm)]
        svgT.extend(['L'+str(pTxli*pm + Offset)+','+str(pTyli*pm + Offset + y_offset * pm) for pTxli, pTyli in zip(top_x, top_y)])
        svgB = ['L'+str(pBxli*pm + Offset)+','+str(pByli*pm + Offset + y_offset * pm) for pBxli, pByli in zip(bottom_x, bottom_y)]
        svgTList.append(' '.join(svgT)+' '.join(reversed(svgB)) +' z')
        y_offset += max(max(top_y), max(bottom_y)) - min(min(top_y), min(bottom_x))
        max_x = max(max_x, max(top_x) - min(top_x))

    with open(file_name, "w") as f:
        strFile="<?xml version='1.0' encoding='UTF-8' standalone='no'?>\r\n<!DOCTYPE svg PUBLIC '-//W3C//DTD SVG 1.1//EN' 'http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd'>\r\n<svg width='{}pt' height='{}pt' viewBox='0 0 {} {}' overflow='visible' version='1.1' xmlns='http://www.w3.org/2000/svg'>\r\n".format(max_x + 10, y_offset, max_x + 10, y_offset)
        f.write(strFile)

        y_offset = 0

        for list_index in range(len(top_x_list)):
            top_x, top_y, bottom_x, bottom_y = top_x_list[list_index], top_y_list[list_index], bottom_x_list[list_index], bottom_y_list[list_index], 

            joint_x, joint_y, joint_index_list = joint_x_list[list_index], joint_y_list[list_index], joint_index_list_list[list_index]

            ribbon_index, joint_angle_list, joint_direction_list = ribbon_index_list[list_index], joint_angle_list_list[list_index], all_joint_direction_list[list_index]

            upper_joint_x, upper_joint_y, lower_joint_x, lower_joint_y = upper_joint_x_list[list_index], upper_joint_y_list[list_index], lower_joint_x_list[list_index], lower_joint_y_list[list_index]

            f.write('<g>\n')
            y_offset += 0 - min(top_y) + 10
            for index, (pCirx, pCirY, j_index, j_angle, j_direction, upper_jx, upper_jy, lower_jx, lower_jy) in enumerate(zip(joint_x, joint_y, joint_index_list, joint_angle_list, joint_direction_list, upper_joint_x, upper_joint_y, lower_joint_x, lower_joint_y)):
                # Draw circle or slits for crossing
                if (not use_slit) or (slit_orientation == []):
                    f.write('<circle cx="'+str(pCirx*pm + Offset)+'" cy="'+str(pCirY*pm + Offset+ y_offset * pm)+'" r="'+str(pm*rHole)+'" stroke="' + color_hole+'" stroke-width="'+str(stroke)+'" fill-opacity="0" />\r\n')
                else:
                    x2, y2 = lower_jx, lower_jy
                    if slit_orientation[list_index][index] == 1:
                        x2, y2 = upper_jx, upper_jy
                    f.write('<line x1="'+str(pCirx*pm + Offset)+'" y1="'+str(pCirY*pm + Offset+ y_offset * pm)+'" x2="'+str(x2*pm + Offset)+'" y2="'+str(y2*pm + Offset+ y_offset * pm)+'" stroke="' + color_hole+'" stroke-width="'+str(thickness)+'" fill-opacity="1" />\r\n')
                # Write the crossing index next to the circle or slits.
                x = pCirx*pm + Offset + j_direction[0] * rHole * pm * 4
                y = pCirY*pm + Offset + y_offset * pm + j_direction[1] * rHole * pm * 4
                f.write('<text x="{}" y="{}" text-anchor="middle" dominant-baseline="central" transform="rotate({} {} {})" fill="none" stroke="blue" stroke-width="0.05" font-size="{}">{}</text>\r\n'.format(x, y, -j_angle / np.pi * 180, x, y, rHole*pm*3, j_index))

            # Write the ribbon index at the end of the ribbon.
            x = joint_x[0]*pm + Offset - joint_direction_list[0][0] * rHole * pm * 4
            y = joint_y[0]*pm + Offset + y_offset * pm - joint_direction_list[0][1] * rHole * pm * 4
            f.write('<text x="{}" y="{}" text-anchor="middle" dominant-baseline="central" transform="rotate({} {} {})" fill="none" stroke="blue" stroke-width="0.05" font-size="{}">{}</text>\r\n'.format(x, y, -joint_angle_list[0] / np.pi * 180, x, y, rHole*pm*3, ribbon_index))
            y_offset += max(max(top_y), max(bottom_y)) - min(min(top_y), min(bottom_x))
            f.write('</g>\n')

        # Write the lines and curves
        for svgTLi in svgTList:
            f.write(pre + svgTLi + post +"\r\n")

        f.write("</svg>")

def get_all_curve_pattern(linkage, thickness, rod_resolution, RIBBON_NAME, image_type = 'svg', flip_angles = False, iteration_index = 0, select_ribbon_index = [], target_ribbon_width = 5, rest = True, bending_axis = 0, use_slit = False, slit_orientation = []):
    ribbons = order_segments_by_ribbons(linkage)
    all_ribbon_angle, all_ribbon_edge_len, all_ribbon_num_seg, all_ribbon_widths, all_joint_index, _ = get_turning_angle_and_length_from_ordered_rods(ribbons, linkage, rest = rest, bending_axis = bending_axis)
    if flip_angles:
        all_ribbon_angle = [[w * -1 for w in row] for row in all_ribbon_angle]

     # np.savez('free_linkage_ribbon_angle_edge_len.npz', ribbons = ribbons, all_ribbon_angle = all_ribbon_angle, all_ribbon_edge_len = all_ribbon_edge_len)
    # Parameters for fabrication
    longest_rod_length = max([sum(edge_len) for edge_len in all_ribbon_edge_len])
    desired_longest_length = all_ribbon_num_seg[0] * 45
    # scale = desired_longest_length / longest_rod_length

    min_width = min([min(row) for row in all_ribbon_widths])
    print('Min Width ', min_width)
    scale = target_ribbon_width / min_width
    print('Scale: ', scale)
    # print(longest_rod_length, desired_longest_length, scale)
    all_ribbon_widths = [[w * scale / 2 for w in row] for row in all_ribbon_widths]

    h = thickness * scale
    top_x_list, top_y_list, bottom_x_list, bottom_y_list, joint_x_list, joint_y_list, joint_index_list, ribbon_index_list, all_joint_angle_list, all_joint_direction_list, upper_joint_x_list, upper_joint_y_list, lower_joint_x_list, lower_joint_y_list = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    if select_ribbon_index == []:
        select_ribbon_index = range(len(all_ribbon_angle))

    if not os.path.exists(RIBBON_NAME):
        os.makedirs(RIBBON_NAME)
    for ribbon_index in select_ribbon_index:
        top_x, top_y, bottom_x, bottom_y, joint_x, joint_y, joints_angle, joints_direction, upper_joints_xs, upper_joints_ys, lower_joints_xs, lower_joints_ys = get_laser_cutting_pattern_data(ribbons, all_ribbon_angle, all_ribbon_edge_len, all_ribbon_num_seg, scale, h, all_ribbon_widths, rod_resolution, ribbon_index = ribbon_index)
        top_x_list.append(top_x) 
        top_y_list .append(top_y) 
        bottom_x_list.append(bottom_x) 
        bottom_y_list.append(bottom_y) 
        joint_x_list.append(joint_x) 
        joint_y_list.append(joint_y) 
        ribbon_index_list.append(ribbon_index) 
        all_joint_angle_list.append(joints_angle)
        all_joint_direction_list.append(joints_direction)
        upper_joint_x_list.append(upper_joints_xs) 
        upper_joint_y_list.append(upper_joints_ys) 
        lower_joint_x_list.append(lower_joints_xs) 
        lower_joint_y_list.append(lower_joints_ys) 
        if image_type == 'png':
            plt.axes().set_aspect('equal')
            plt.plot(top_x, top_y, '-', label='shifted up', linewidth = 2, color = 'black')
            plt.plot(bottom_x, bottom_y, '-', label='shifted down', linewidth = 2, color = 'black')
            plt.scatter(joint_x, joint_y, s = 0.5, facecolors='none', edgecolors='black')
            plt.tight_layout()
            plt.savefig('./{}/{}_{}.png'.format(RIBBON_NAME, ribbon_index, iteration_index), format='png', dpi = 300)

    if image_type == 'svg':
        get_all_curve_svg('./{}/iter{}_ribbons{}.svg'.format(RIBBON_NAME, iteration_index, '-'.join([str(x) for x in select_ribbon_index])), top_x_list, top_y_list, bottom_x_list, bottom_y_list, joint_x_list, joint_y_list, all_joint_index, ribbon_index_list, all_joint_angle_list, all_joint_direction_list, upper_joint_x_list, upper_joint_y_list, lower_joint_x_list, lower_joint_y_list, h, use_slit, slit_orientation)
    
def write_per_ribbon_svg(linkage, thickness, rod_resolution, RIBBON_NAME, image_type = 'svg', flip_angles = False, iteration_index = 0, select_ribbon_index = [], target_ribbon_width = 5, rest = True, bending_axis = 0, use_slit = False, slit_orientation = []):
    ribbons = order_segments_by_ribbons(linkage)
    all_ribbon_angle, all_ribbon_edge_len, all_ribbon_num_seg, all_ribbon_widths, all_joint_index, _ = get_turning_angle_and_length_from_ordered_rods(ribbons, linkage, rest = rest, bending_axis = bending_axis)
    if flip_angles:
        all_ribbon_angle = [[w * -1 for w in row] for row in all_ribbon_angle]

     # np.savez('free_linkage_ribbon_angle_edge_len.npz', ribbons = ribbons, all_ribbon_angle = all_ribbon_angle, all_ribbon_edge_len = all_ribbon_edge_len)
    # Parameters for fabrication
    longest_rod_length = max([sum(edge_len) for edge_len in all_ribbon_edge_len])
    desired_longest_length = all_ribbon_num_seg[0] * 45
    # scale = desired_longest_length / longest_rod_length

    min_width = min([min(row) for row in all_ribbon_widths])
    print('Min Width ', min_width)
    scale = target_ribbon_width / min_width
    print('Scale: ', scale)
    # print(longest_rod_length, desired_longest_length, scale)
    all_ribbon_widths = [[w * scale / 2 for w in row] for row in all_ribbon_widths]

    h = thickness * scale
    top_x_list, top_y_list, bottom_x_list, bottom_y_list, joint_x_list, joint_y_list, joint_index_list, ribbon_index_list, all_joint_angle_list, all_joint_direction_list, upper_joint_x_list, upper_joint_y_list, lower_joint_x_list, lower_joint_y_list = [], [], [], [], [], [], [], [], [], [], [], [], [], []
    if select_ribbon_index == []:
        select_ribbon_index = range(len(all_ribbon_angle))

    if not os.path.exists(RIBBON_NAME):
        os.makedirs(RIBBON_NAME)
    for ribbon_index in select_ribbon_index:
        top_x, top_y, bottom_x, bottom_y, joint_x, joint_y, joints_angle, joints_direction, upper_joints_xs, upper_joints_ys, lower_joints_xs, lower_joints_ys = get_laser_cutting_pattern_data(ribbons, all_ribbon_angle, all_ribbon_edge_len, all_ribbon_num_seg, scale, h, all_ribbon_widths, rod_resolution, ribbon_index = ribbon_index)
        top_x_list.append(top_x) 
        top_y_list .append(top_y) 
        bottom_x_list.append(bottom_x) 
        bottom_y_list.append(bottom_y) 
        joint_x_list.append(joint_x) 
        joint_y_list.append(joint_y) 
        ribbon_index_list.append(ribbon_index) 
        all_joint_angle_list.append(joints_angle)
        all_joint_direction_list.append(joints_direction)
        upper_joint_x_list.append(upper_joints_xs) 
        upper_joint_y_list.append(upper_joints_ys) 
        lower_joint_x_list.append(lower_joints_xs) 
        lower_joint_y_list.append(lower_joints_ys) 

    if not os.path.exists('ribbon_rest_state'):
            os.makedirs('ribbon_rest_state')
    for ribbon_index in select_ribbon_index:
        get_all_curve_svg('./ribbon_rest_state/{}_{}.svg'.format(RIBBON_NAME, ribbon_index), [top_x_list[ribbon_index]], [top_y_list[ribbon_index]], [bottom_x_list[ribbon_index]], [bottom_y_list[ribbon_index]], [joint_x_list[ribbon_index]], [joint_y_list[ribbon_index]], [all_joint_index[ribbon_index]], [ribbon_index_list[ribbon_index]], [all_joint_angle_list[ribbon_index]], [all_joint_direction_list[ribbon_index]], [upper_joint_x_list[ribbon_index]], [upper_joint_y_list[ribbon_index]], [lower_joint_x_list[ribbon_index]], [lower_joint_y_list[ribbon_index]], h, use_slit, slit_orientation)



