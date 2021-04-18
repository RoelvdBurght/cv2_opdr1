import numpy as np
import open3d as o3d
import scipy as sp
import scipy.io
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse


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

def closest_point_brute_force(matrix, target, sampling_method='all', sample_size=0.5, 
								uniform_indices=None):
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
	all_sampled_indices = []
	# Loop through all splits, computing each point-combinations' distance
	for i in tqdm(range(splits)):
		
		# Use all points
		if sampling_method == 'all':
			sub_matrix = matrix[i*matrix_len:(i+1)*matrix_len]

		# Use random subsampling
		if sampling_method == 'random':
			sub_matrix = matrix[i*matrix_len:(i+1)*matrix_len]
			sample_size = int(sample_size * len(sub_matrix))
			indices = range(len(sub_matrix))
			sampled_incdices = np.random.choice(indices, sample_size)
			sub_matrix = sub_matrix[sampled_incdices]

			# Translate the sub_matrix indices back to the original matrix indices
			all_sampled_indices.extend([s+i*matrix_len for s in sampled_incdices])
		
		# Use uniform subsampling
		if sampling_method == 'uniform':
			indices = uniform_indices[i*matrix_len:(i+1)*matrix_len]
			sub_matrix = matrix[indices]
			all_sampled_indices.extend(indices)


		dist = sp.spatial.distance_matrix(sub_matrix, target)

		target_indices = np.argmin(dist, axis=1)
		closest_points_indices = np.append(closest_points_indices, target_indices)
		

	# Also return sampled indices for rms calculation, in case of uniform sampling
	if sampling_method == 'random' or sampling_method == 'uniform' :
		return closest_points_indices.astype(int), all_sampled_indices
	
	return closest_points_indices.astype(int), list(range(len(matrix)))

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



############################
#   Load Data              #
############################
##  Load source (pcd) and target image
def ICP(args):
	data_path = '../Data/data/'
	pcd = o3d.io.read_point_cloud(f"{data_path}0000000000.pcd")
	target = o3d.io.read_point_cloud(f"{data_path}0000000001.pcd")

	## convert into ndarray
	pcd_arr = np.asarray(pcd.points)
	target = np.asarray(target.points)

	# ***  you need to clean the point cloud using a threshold ***
	source = remove_outliers(pcd_arr)
	target = remove_outliers(target)


	pcd_dict = sp.io.loadmat("../Data/source.mat")
	pcd_arr = pcd_dict['source'].T
	target_dict = sp.io.loadmat("../Data/target.mat")
	target = target_dict['target'].T
	source = remove_outliers(pcd_arr)


	## visualization from ndarray
	# vis_pcd = o3d.geometry.PointCloud()
	# vis_pcd.points = o3d.utility.Vector3dVector(target)
	# o3d.visualization.draw_geometries([vis_pcd])

	############################
	#     ICP                  #
	############################

	sampling_method = args.sampling_method
	sample_size = args.sampling_size
	
	# For uniform sampling determine the points to use
	sampled_incdices = None
	if sampling_method == 'uniform':
		indices = range(len(source))
		sample_size = int(sample_size * len(source))
		sampled_incdices = np.random.choice(indices, sample_size)
	print(sampled_incdices.shape)
	# Find the closest point for each point in A1 based on A2 using brute-force approach
	closest_points_indices, source_indices = closest_point_brute_force(source, target,
						sampling_method=sampling_method, sample_size=sample_size,
						uniform_indices=sampled_incdices)

	# Compute RMS
	rms = calc_rms(source[source_indices], target[closest_points_indices])
	print("First RMS: {}".format(rms))

	# Loop till convergence or max iteration
	max_iteration = 50
	convergence = -0.1

	for i in range(max_iteration):
		
		# Refine R and t using SVD, and transform pointcloud
		R, t = calc_R_and_t(source[source_indices], target[closest_points_indices])
		source = source @ R + t

		new_rms = calc_rms(source[source_indices], target[closest_points_indices])
		print("RMS iter{}: {}".format(i, new_rms))
		# Break if converged
		if (new_rms - rms) / rms > convergence and i > 0:
			break

		rms = new_rms

		# Compute new closest points and RMS
		closest_points_indices, source_indices = closest_point_brute_force(source, target, 
							sampling_method=sampling_method, sample_size=sample_size,
							uniform_indices=sampled_incdices)



if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Training params
    parser.add_argument('--sampling_method', type=str, default='all',
    					choices=['all', 'random', 'uniform'],
                        help='sampling method to use')
    parser.add_argument('--sampling_size', type=float, default=0.5,
                        help='size of the sample')
    
    args = parser.parse_args()
    ICP(args)
############################
#   Merge Scene            #
############################

#  Estimate the camera poses using two consecutive frames of given data.

#  Estimate the camera pose and merge the results using every 2nd, 4th, and 10th frames.

#  Iteratively merge and estimate the camera poses for the consecutive frames.



############################
#  Additional Improvements #
############################
