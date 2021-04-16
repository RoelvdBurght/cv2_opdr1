import numpy as np
import open3d as o3d
import scipy as sp
import scipy.io
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

######                                                           ######
##       notice: You don't need to strictly follow the steps         ##
######                                                           ######

######################################
##        Functions for ICP         ##
######################################

def remove_outliers(matrix, threshold=2):
	# Removes outliers, all points further than 2 meters away
	matrix = matrix[matrix[:, 2] < threshold]
	return matrix

def closest_point_brute_force(matrix, target):
	"""
		Computes the points in target closest to each point in matrix, using
		sp.spatial.distance_matrix() for the distance computation.
		The computation is divided into splits, because of the size of both matrices,
		and due to the fact that we compute the distances between each point combination.
	"""

	# Initialize variables
	splits = 100
	matrix_len = int(len(matrix)/splits)
	closest_points_indices = np.array([])

	# Loop through all splits, computing each point-combinations' distance
	for i in tqdm(range(splits+1)):
		sub_matrix = matrix[i*matrix_len:(i+1)*matrix_len]
		dist = sp.spatial.distance_matrix(sub_matrix, target)
		target_indices = np.argmin(dist, axis=1)
		closest_points_indices = np.append(closest_points_indices, target_indices)

	# Return the closest points from the indices
	closest_points = target[closest_points_indices.astype(int)]
	return closest_points

def calc_rms(matrix, closest_points):
	# Computes the RMS
	diff = matrix - closest_points
	summed_powered_diff = np.sum(np.power(diff, 2))
	rms = np.sqrt(summed_powered_diff/len(matrix))
	return rms

def calc_R_and_t(matrix, closest_points):
	# Compute centroids, weights are set to 1
	p = np.mean(matrix, axis=0)
	q = np.mean(closest_points, axis=0)

	# Compute centred vectors
	x = matrix - p
	y = closest_points - q

	# Compute dxd covariance matrix
	X = x.T
	Y = y.T
	S = X @ Y.T

	# Compute SVD
	U, Sigma, V = np.linalg.svd(S)
	V = V.T

	# Compute R
	det = np.linalg.det(V @ U.T)
	temp = np.eye(len(V))
	temp[len(V)-1, len(V)-1] = det
	R = V @ temp @ U.T

	# Compute optimal translation
	t = q - R @ p

	return R.T, t

def calc_metrics():



############################
#   Load Data              #
############################
##  Load source (pcd) and target image
data_path = '../Data/data/'
pcd = o3d.io.read_point_cloud(f"{data_path}0000000000.pcd")
target = o3d.io.read_point_cloud(f"{data_path}0000000001.pcd")

## convert into ndarray
pcd_arr = np.asarray(pcd.points)
target = np.asarray(target.points)

# ***  you need to clean the point cloud using a threshold ***
source = remove_outliers(pcd_arr)
target = remove_outliers(target)

## visualization from ndarray
# vis_pcd = o3d.geometry.PointCloud()
# vis_pcd.points = o3d.utility.Vector3dVector(target)
# o3d.visualization.draw_geometries([vis_pcd])

############################
#     ICP                  #
############################

# Find the closest point for each point in A1 based on A2 using brute-force approach
closest_points = closest_point_brute_force(source, target)

# Compute RMS
rms = calc_rms(source, closest_points)
print("First RMS: {}".format(rms))

# Loop till convergence or max iteration
max_iteration = 5
convergence = -0.1

for i in range(max_iteration):
	t0 = time.time()
	# Refine R and t using SVD, and transform pointcloud
	R, t = calc_R_and_t(source, closest_points)
	source = source @ R + t

	# Compute new closest points and RMS
	closest_points = closest_point_brute_force(source, target)
	new_rms = calc_rms(source, closest_points)
	print("RMS iter{}: {}".format(i, new_rms))
	iter_time = 
	# Break if converged
	if (new_rms - rms) / rms > convergence:
		break

	rms = new_rms

############################
#   Merge Scene            #
############################

#  Estimate the camera poses using two consecutive frames of given data.

#  Estimate the camera pose and merge the results using every 2nd, 4th, and 10th frames.

#  Iteratively merge and estimate the camera poses for the consecutive frames.



############################
#  Additional Improvements #
############################
