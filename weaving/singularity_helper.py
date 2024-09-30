import numpy.linalg as la
import numpy as np
import os
def get_radius(length, num_sides):
    theta = np.pi * 2 / num_sides
    beta = (np.pi - theta) / 2
    radius = length * np.sin(beta) / np.sin(theta)
    return radius

def generate_biaxial_grid(grid_size):
    unit_width = 1
    vertices = []
    edges = []

    for i in range(grid_size):
        for j in range(grid_size):
            vertices.append([i, j, 0])

    for i in range(grid_size-1):
        for j in range(grid_size):
            edges.append([i, i + 1])
            edges.append([j, j + 1])
    biaxial_path = 'biaxial'
    if not os.path.exists(biaxial_path):
        os.makedirs(biaxial_path)

    with open ('{}/{}_grid.obj'.format(biaxial_path, grid_size), 'w') as f:
        for point in vertices:
            f.write('v {} {} {}\n'.format(point[0], point[1], point[2]))
        for edge in edges:
            f.write('l {} {}\n'.format(edge[0]+1, edge[1]+1))

def generate_polygon_module(poly_size, flat_start = True):
    polygon = []
    radius = get_radius(edge_length, poly_size)
    for i in range(poly_size):
        theta = 2 * np.pi / poly_size * i
        point = np.array([radius * np.sin(theta), radius * np.cos(theta), 0])
        polygon.append(point)

    tips = []
    if flat_start:
        for i in range(poly_size):
            mid = (polygon[i-1] + polygon[i]) / 2
            height = la.norm(polygon[i-1] - polygon[i]) * np.sqrt(3) / 2
            tips.append(mid + mid / la.norm(mid) * height)
    else:
        for i in range(poly_size):
            mid = (polygon[i-1] + polygon[i]) / 2
            height = np.array([0, 0, la.norm(polygon[i-1] - polygon[i]) * np.sqrt(3) / 2])
            tips.append(mid + height)
    
    point_list = polygon + tips
    edge_list = [(i + 1, i + 2) for i in range(poly_size - 1)] + [(poly_size, 1)] + [(i + 1, poly_size + i + 2) for i in range(poly_size - 1)]  + [(poly_size, poly_size + 1)] + [(i + 2, poly_size + i + 2) for i in range(poly_size - 1)] + [(1, poly_size + 1)]
    polygon_path = 'polygon'
    if not os.path.exists(polygon_path):
        os.makedirs(polygon_path)

    with open ('{}/{}_gon.obj'.format(polygon_path, poly_size), 'w') as f:
        for point in point_list:
            f.write('v {} {} {}\n'.format(point[0], point[1], point[2]))
        for edge in edge_list:
            f.write('l {} {}\n'.format(edge[0], edge[1]))
def generate_singularity_topology(polygon_size = 5):
    vertices = []
    edges = []
    radius = get_radius(1, polygon_size)
    for i in range(polygon_size):
        theta = 2 * np.pi / polygon_size * i
        point = np.array([radius * np.sin(theta), radius * np.cos(theta), 0])
        vertices.append(point)
        edges.append([i, (i + 1)%polygon_size])
    for i in range(polygon_size):
        theta = 2 * np.pi / polygon_size * i + np.pi / polygon_size
        point = np.array([2 * radius * np.sin(theta), 2 * radius * np.cos(theta), 0])
        vertices.append(point)
        edges.append([i + polygon_size, i])
        edges.append([i + polygon_size, (i+1)%polygon_size])
    for i in range(polygon_size):
        theta_1 = 2 * np.pi / polygon_size * i + 0.5 *  np.pi / polygon_size
        theta_2 = 2 * np.pi / polygon_size * i + 1.5 * np.pi / polygon_size
        point = np.array([3 * radius * np.sin(theta_1), 3 * radius * np.cos(theta_1), 0])
        vertices.append(point)
        point = np.array([3 * radius * np.sin(theta_2), 3 * radius * np.cos(theta_2), 0])
        vertices.append(point)
        edges.append([i * 2 + polygon_size * 2, i + polygon_size])
        edges.append([i * 2 + polygon_size * 2 + 1, i + polygon_size])
        edges.append([i * 2 + polygon_size * 2, i * 2 + polygon_size * 2 + 1])

    for i in range(polygon_size):
        theta = 2 * np.pi / polygon_size * i
        point = np.array([4 * radius * np.sin(theta), 4 * radius * np.cos(theta), 0])
        vertices.append(point)
        if i == 0:
            edges.append([i + polygon_size * 4, polygon_size * 2 - 1 + polygon_size * 2])
            edges.append([i + polygon_size * 4, i * 2 + polygon_size * 2])
        else:
            edges.append([i + polygon_size * 4, i * 2 - 1 + polygon_size * 2])
            edges.append([i + polygon_size * 4, i * 2 + polygon_size * 2])

    for i in range(polygon_size):
        theta_1 = 2 * np.pi / polygon_size * i + 0.5 *  np.pi / polygon_size
        theta_2 = 2 * np.pi / polygon_size * i - 0.5 * np.pi / polygon_size
        point = np.array([5 * radius * np.sin(theta_1), 5 * radius * np.cos(theta_1), 0])
        vertices.append(point)
        point = np.array([5 * radius * np.sin(theta_2), 5 * radius * np.cos(theta_2), 0])
        vertices.append(point)
        edges.append([i * 2 + polygon_size * 5, i + polygon_size * 4])
        edges.append([i * 2 + polygon_size * 5 + 1, i + polygon_size * 4])
        
        edges.append([i * 2 + polygon_size * 5, i * 2 + polygon_size * 2])
        if i == 0:
            edges.append([i * 2 + polygon_size * 5 + 1, polygon_size * 4-1])
        else:
            edges.append([i * 2 + polygon_size * 5 + 1, i * 2 + polygon_size * 2 - 1])

    for i in range(polygon_size):
        theta_1 = 2 * np.pi / polygon_size * i + 0.25 *  np.pi / polygon_size
        theta_2 = 2 * np.pi / polygon_size * i + 0.5 * np.pi / polygon_size
        theta_3 = 2 * np.pi / polygon_size * i - 0.25 * np.pi / polygon_size
        theta_4 = 2 * np.pi / polygon_size * i - 0.5 * np.pi / polygon_size
        point = np.array([7 * radius * np.sin(theta_1), 7 * radius * np.cos(theta_1), 0])
        vertices.append(point)
        point = np.array([7 * radius * np.sin(theta_2), 7 * radius * np.cos(theta_2), 0])
        vertices.append(point)
        point = np.array([7 * radius * np.sin(theta_3), 7 * radius * np.cos(theta_3), 0])
        vertices.append(point)
        point = np.array([7 * radius * np.sin(theta_4), 7 * radius * np.cos(theta_4), 0])
        vertices.append(point)
        
        edges.append([i * 4 + polygon_size * 7, i * 2 + polygon_size * 5])
        edges.append([i * 4 + polygon_size * 7 + 1, i * 2 + polygon_size * 5])
        
        if i == 0:
            edges.append([i * 4 + polygon_size * 7 + 2, polygon_size * 5+1])
            edges.append([i * 4 + polygon_size * 7 + 3, polygon_size * 5+1])
        else:
            edges.append([i * 4 + polygon_size * 7 + 2, i * 2 + polygon_size * 5+1])
            edges.append([i * 4 + polygon_size * 7 + 3, i * 2 + polygon_size * 5+1])
        


    singularity_path = 'singularity_patches'
    if not os.path.exists(singularity_path):
        os.makedirs(singularity_path)

    with open ('{}/{}_gon.obj'.format(singularity_path, polygon_size), 'w') as f:
        for point in vertices:
            f.write('v {} {} {}\n'.format(point[0], point[1], point[2]))
        for edge in edges:
            f.write('l {} {}\n'.format(edge[0]+1, edge[1]+1))