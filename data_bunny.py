#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import h5py
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import minkowski


def download_bunny():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'bunny')):
        www = 'http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz'
        bunny_file = os.path.basename(www)
        os.system(f'wget {www} -O {bunny_file} --no-check-certificate')
        print(f"Downloaded '{bunny_file}'.")
        if bunny_file.endswith('.tar') or bunny_file.endswith('.tar.gz'):
            os.system(f'tar -xvf {bunny_file} ')
            print(f"Extraction successful.")
        else:
            print("The file is not a recognized tarball. No extraction performed.")
        os.system(f'rm {bunny_file}')
        


def load_bunny_data():
    download_bunny()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    
    ply_path = os.path.join(DATA_DIR, 'bunny', 'reconstruction/bun_zipper.ply')
    mesh = o3d.io.read_triangle_mesh(ply_path)

    if mesh.is_empty():
        print("Failed to load the mesh.")
        return

    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=1000)
    bunny_data = np.asarray(pcd.points)

    # Save the point cloud in PCD format
    pcd_filename = os.path.splitext(ply_path)[0] + ".pcd"
    o3d.io.write_point_cloud(pcd_filename, pcd)
    
    print(bunny_data.shape)  # (1000, 3)
        
    def random_rotation_matrix():
        theta = np.random.uniform(0, np.pi * 2)
        phi = np.random.uniform(0, np.pi * 2)
        z = np.random.uniform(0, np.pi * 2)

        rot_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ])

        rot_y = np.array([
            [np.cos(phi), 0, np.sin(phi)],
            [0, 1, 0],
            [-np.sin(phi), 0, np.cos(phi)]
        ])

        rot_z = np.array([
            [np.cos(z), -np.sin(z), 0],
            [np.sin(z), np.cos(z), 0],
            [0, 0, 1]
        ])

        rotation_matrix = np.dot(np.dot(rot_z, rot_y), rot_x)
        return rotation_matrix

    # 创建一个大数组，用于存储增强后的数据
    all_data = np.zeros((200, 1000, 3))

    # 对每个副本应用随机旋转
    for i in range(200):
        rotation_matrix = random_rotation_matrix()
        all_data[i] = np.dot(bunny_data, rotation_matrix.T)  # 应用旋转

    print(all_data.shape)

    return all_data 

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)      #抖动，同形状noise matrix
    return pointcloud


def farthest_subsample_points(pointcloud1, pointcloud2, num_subsampled_points=768):
    pointcloud1 = pointcloud1.T
    pointcloud2 = pointcloud2.T
    num_points = pointcloud1.shape[0]

    # 确保点云为二维数组
    if pointcloud1.ndim == 1:
        pointcloud1 = pointcloud1.reshape(1, -1)
    if pointcloud2.ndim == 1:
        pointcloud2 = pointcloud2.reshape(1, -1) 

    nbrs1 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud1)
    random_p1 = np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 1, -1])
    idx1 = nbrs1.kneighbors(random_p1, return_distance=False).reshape((num_subsampled_points,))
    nbrs2 = NearestNeighbors(n_neighbors=num_subsampled_points, algorithm='auto',
                             metric=lambda x, y: minkowski(x, y)).fit(pointcloud2)
    random_p2 = random_p1 #np.random.random(size=(1, 3)) + np.array([[500, 500, 500]]) * np.random.choice([1, -1, 2, -2])
    idx2 = nbrs2.kneighbors(random_p2, return_distance=False).reshape((num_subsampled_points,))
    return pointcloud1[idx1, :].T, pointcloud2[idx2, :].T


class StanfordBunny(Dataset):
    def __init__(self, num_points, num_subsampled_points, gaussian_noise=False, rot_factor=4):
        super(StanfordBunny, self).__init__()
        self.data = load_bunny_data()
        self.num_points = num_points
        self.num_subsampled_points = num_subsampled_points
        if num_points != num_subsampled_points:
            self.subsampled = True
        else:
            self.subsampled = False
        self.gaussian_noise = gaussian_noise
        self.rot_factor = rot_factor

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        np.random.seed(item)
        anglex = np.random.uniform() * np.pi / self.rot_factor
        angley = np.random.uniform() * np.pi / self.rot_factor
        anglez = np.random.uniform() * np.pi / self.rot_factor
        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        Rx = np.array([[1, 0, 0],
                        [0, cosx, -sinx],
                        [0, sinx, cosx]])
        Ry = np.array([[cosy, 0, siny],
                        [0, 1, 0],
                        [-siny, 0, cosy]])
        Rz = np.array([[cosz, -sinz, 0],
                        [sinz, cosz, 0],
                        [0, 0, 1]])
        R_ab = Rx.dot(Ry).dot(Rz)
        R_ba = R_ab.T
        translation_ab = np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5),
                                   np.random.uniform(-0.5, 0.5)])
        translation_ba = -R_ba.dot(translation_ab)

        pointcloud1 = pointcloud.T

        rotation_ab = Rotation.from_euler('zyx', [anglez, angley, anglex])
        pointcloud2 = rotation_ab.apply(pointcloud1.T).T + np.expand_dims(translation_ab, axis=1)

        euler_ab = np.asarray([anglez, angley, anglex])
        euler_ba = -euler_ab[::-1]

        pointcloud1 = np.random.permutation(pointcloud1.T).T
        pointcloud2 = np.random.permutation(pointcloud2.T).T

        if self.gaussian_noise:
            pointcloud1 = jitter_pointcloud(pointcloud1)
            pointcloud2 = jitter_pointcloud(pointcloud2)

        if self.subsampled:
            # print(pointcloud1.shape)
            # print(pointcloud2.shape)
            pointcloud1, pointcloud2 = farthest_subsample_points(pointcloud1, pointcloud2,
                                                                 num_subsampled_points=self.num_subsampled_points)

        return pointcloud1.astype('float32'), pointcloud2.astype('float32'), R_ab.astype('float32'), \
               translation_ab.astype('float32'), R_ba.astype('float32'), translation_ba.astype('float32'), \
               euler_ab.astype('float32'), euler_ba.astype('float32')

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    print('hello world')
