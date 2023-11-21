import open3d as o3d
import numpy as np
import os
import tarfile
import wget
import copy
from util import visualize_pcs

url = 'http://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz'
filename = 'bunny.tar.gz'
extract_dir = '/home/ytanaz/comp_5212/ML_Prnet_JL/ML_Prnet/data'              #only change this into your local filepath
# extract_dir = 'D:/UST/Research/ML_Prnet/data'              #only change this into your local filepath
ply_path = '/bunny/reconstruction/bun_zipper.ply'


def download_and_extract(url, filename, extract_dir):
    """Download and extract a dataset from a URL."""
    try:
        if not os.path.exists('bunny.tar.gz'):
            # Download the file
            wget.download(url, out=filename)
            print(f"\nDownloaded '{filename}'.")
        else: 
            print('File exists.')
        
        # Extract the tarball content
        with tarfile.open(filename) as tar:
            tar.extractall(path=extract_dir)
        print(f"Extraction successful to directory: {extract_dir}")

        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def load(ply_path, color_rgb, points_num):
    """Load a PLY file and visualize it with a uniform color."""
    mesh = o3d.io.read_triangle_mesh(ply_path)

    if mesh.is_empty():
        print("Failed to load the mesh.")
        return

    mesh.compute_vertex_normals()
    pcd = mesh.sample_points_poisson_disk(number_of_points=points_num)

    # Set the color for all points in the point cloud
    num_points = np.asarray(pcd.points).shape[0]
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color_rgb, (num_points, 1)))
    
    # Save the point cloud in PCD format
    pcd_filename = os.path.splitext(ply_path)[0] + ".pcd"
    o3d.io.write_point_cloud(pcd_filename, pcd)

    return pcd 


if __name__ == '__main__':
    # download and visualize the init bunny 
    if download_and_extract(url, filename, extract_dir):
        ply_path = extract_dir + ply_path
        pcd = load(ply_path, [1, 0, 0], 1000)  # Red

    # Make a copy of the point cloud for transformation
    pcd_transformed = copy.deepcopy(pcd)

    # Define a random rotation matrix (3x3)
    random_rotation = o3d.geometry.get_rotation_matrix_from_xyz(np.random.uniform(0, 2 * np.pi, (3,)))

    # Define a random translation vector
    random_translation = np.random.uniform(-1, 1, (3,))

    # Create a transformation matrix (4x4) from the rotation matrix and translation vector
    transformation_matrix = np.identity(4)  # Start with an identity matrix
    transformation_matrix[:3, :3] = random_rotation  # Add rotation
    transformation_matrix[:3, 3] = random_translation  # Add translation

    # Apply the transformation to the copy of the point cloud
    pcd_transformed.transform(transformation_matrix)

    # Optionally, assign different colors to the original and transformed point clouds to distinguish them
    pcd.paint_uniform_color([1, 0, 0])  # Red for the original point cloud
    pcd_transformed.paint_uniform_color([0, 1, 0])  # Green for the transformed point cloud

    print('Get PCD and transformed PCD.')
    # Visualize the original and transformed point clouds together 
    # o3d.visualization.draw_geometries([pcd, pcd_transformed])


    visualize_pcs(pcd, pcd_transformed, 'unit_test')