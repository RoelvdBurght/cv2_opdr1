import numpy as np
import open3d as o3d
import scipy as sp
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

######                                                           ######
##       notice: You don't need to strictly follow the steps         ##
######                                                           ######

def remove_outliers(matrix, threshold=2):
	matrix = matrix[matrix[:, 2] < threshold]
	return matrix

def closest_point_brute_force(matrix, target):
	"""
		Calculate the points in targer closest to each point in matrix.
		Uses sp.spatial.distance_matrix() to do the distance calculation.
		Because of the size of both matrices, and due to the fact that we calculate all
		distances (between all points), the operation is divided into parts.
		Divide the matrix into submatrices, and calculate the distance between the points
		in the submatrix and target.
	"""
	splits = 100
	matrix_len = int(len(matrix)/splits)
	closest_points_indices = np.array([])
	for i in tqdm(range(splits+1)):

		sub_matrix = matrix[i*matrix_len:(i+1)*matrix_len]
		dist = sp.spatial.distance_matrix(sub_matrix, target)
		target_indices = np.argmin(dist, axis=1)
		closest_points_indices = np.append(closest_points_indices, target_indices)

	# # Final distance calc for the last part of the matrix
	# sub_matrix = matrix[(i+1)*matrix_len:]
	# dist = sp.spatial.distance_matrix(sub_matrix, target)
	# target_indices = np.argmin(dist, axis=1)
	# closest_points_indices = np.append(closest_points_indices, target_indices)

	# Get the closest points from the indeces and return
	closest_points = target[closest_points_indices.astype(int)]
	return closest_points

def calc_rms(matrix, closest_points):
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
	print('X', X.shape)

	Y = y.T
	print('Y', Y.shape)

	S = X @ Y.T
	print('S', S.shape)

	# Compute SVD
	U, Sigma, V = np.linalg.svd(S)
	V = V.T

	# Compute R
	det = np.linalg.det(V @ U.T)
	joe = np.eye(len(V))
	joe[len(V)-1, len(V)-1] = det
	R = V @ joe @ U.T

	# Compute optimal translation
	t = q - R @ p

	return R, t

############################
#   Load Data              #
############################
##  example
data_path = '/home/roel/Desktop/master/CV2/cv2_opdr1/Data/data/'
pcd = o3d.io.read_point_cloud(f"{data_path}0000000000.pcd")
target = o3d.io.read_point_cloud(f"{data_path}0000000013.pcd")
# ## convert into ndarray

pcd_arr = np.asarray(pcd.points)
target = np.asarray(target.points)

# ***  you need to clean the point cloud using a threshold ***
pcd_arr_cleaned = remove_outliers(pcd_arr)
target = remove_outliers(target)

# visualization from ndarray
# vis_pcd = o3d.geometry.PointCloud()
# vis_pcd.points = o3d.utility.Vector3dVector(target)
# o3d.visualization.draw_geometries([vis_pcd])

############################
#     ICP                  #
############################

###### 0. (adding noise)

###### 1. initialize R= I , t= 0
R = np.identity(3)
t = 0

###### go to 2. unless RMS is unchanged(<= epsilon)

###### 2. using different sampling methods


###### 3. transform point cloud with R and t
pcd_transformed = pcd_arr_cleaned @ R - t

###### 4. Find the closest point for each point in A1 based on A2 using brute-force approach
closest_points = closest_point_brute_force(pcd_transformed, target)

###### 5. Calculate RMS
rms = calc_rms(pcd_transformed, closest_points)
print('first', rms)

for i in range(5):

	###### 6. Refine R and t using SVD
	R, t = calc_R_and_t(pcd_transformed, closest_points)
	print('before transform', pcd_transformed.shape)
	print(t)
	pcd_transformed = pcd_transformed @ R + t
	print('after transform', pcd_transformed.shape)

	closest_points = closest_point_brute_force(pcd_transformed, target)
	rms = calc_rms(pcd_transformed, closest_points)
	print(rms)

############################
#   Merge Scene            #
############################

#  Estimate the camera poses using two consecutive frames of given data.

#  Estimate the camera pose and merge the results using every 2nd, 4th, and 10th frames.

#  Iteratively merge and estimate the camera poses for the consecutive frames.



############################
#  Additional Improvements #
############################
