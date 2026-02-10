from cmath import nan
import numpy as np
import h5py
import os
from tqdm import tqdm
import sys
from class_util import classes_s3dis, classes_nyu40, classes_kitti, class_to_id, class_to_color_rgb
import itertools
import random
from sklearn.decomposition import PCA
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, adjusted_mutual_info_score
import math
# import networkx as nx
from scipy.cluster.vq import vq, kmeans
import time
# import matplotlib.pyplot as plt
import scipy.special
import glob
import argparse
import torch

import pointcloud_utils as utils

import open3d as o3d


np.random.seed(0)
#   Curvatures may contain nan values.  Using 12 features instead of 13.  Hopefully does not damage results too much.
FEATURE_SIZE = 12
LITE = None
TEST_AREAS = ['1','2','3','4','5','6','scannet']
resolution = 0.1
add_threshold = 0.5
rmv_threshold = 0.5
#   Original cluster_threshold is 10.
cluster_threshold = 10
save_results = False
cross_domain = False
save_id = 0
agg_nmi = []
agg_ami = []
agg_ars = []
agg_prc = []
agg_rcl = []
agg_iou = []
comp_time_analysis = {
	'feature': [],
	'net': [],
	'neighbor': [],
	'inlier': [],
	'current_net' : [],
	'current_neighbor' : [],
	'current_inlier' : [],
	'iter_net' : [],
	'iter_neighbor' : [],
	'iter_inlier' : [],
}

parser = argparse.ArgumentParser()
parser.add_argument('--num_points', type=int, default=512)
parser.add_argument('--downsample', type=float, default=0)
parser.add_argument('--scale', type=float, default=1)
args = parser.parse_args()

if args.scale == 0:
	print("Scale cannot be 0 as this entail zero division.  Exiting.")
	exit()

SCAN_NAME = "data/rgb/20.ply"
RESULTS_DIR = "results"
TRANSFORMER = True
NUM_INLIER_POINT = NUM_NEIGHBOR_POINT = args.num_points


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from ptv3_util import *

net = RegionTransformerPTv3(1, 1, NUM_INLIER_POINT, NUM_NEIGHBOR_POINT, FEATURE_SIZE).to(device)
net.eval()
checkpoint = torch.load("model_checkpoint.pth")
net.load_state_dict(checkpoint['model_state_dict'])

print("Loading pointcloud.")

source_cloud = o3d.io.read_point_cloud(SCAN_NAME)
# if SCAN_NAME[-7:] == ".xyzrgb":
#     source_cloud.colors = o3d.utility.Vector3dVector(np.asarray(source_cloud.colors) / 255)

seed_idx = utils.pick_points(source_cloud)
seed = np.asarray(source_cloud.points)[seed_idx[0], :].squeeze() # (N, 3) array of seed points

print(f"Cloud point count:  {np.asarray(source_cloud.points).shape[0]}")

source_cloud_np = np.asarray(np.concatenate((np.asarray(source_cloud.points), np.asarray(source_cloud.colors)), axis=1))
source_colors = np.asarray(source_cloud.colors)

down_sample = args.downsample

idx_list = None

if down_sample != 0:
	print("Downsampling.")
	cloud, _, idx_list = source_cloud.voxel_down_sample_and_trace(down_sample, source_cloud.get_min_bound(), source_cloud.get_max_bound())
	print(f"Cloud point count after downsample:  {np.asarray(cloud.points).shape[0]}")
else:
	cloud = source_cloud

cloud_center = cloud.get_center()
cloud = cloud.translate(-1*cloud_center, relative=True)

print(f"Cloud min bound:  {cloud.get_min_bound()}")
print(f"Cloud max bound:  {cloud.get_max_bound()}")

scale = args.scale
if scale != 1:
	cloud = cloud.scale(scale, cloud.get_center())
	print(f"Cloud min bound after scale:  {cloud.get_min_bound()}")
	print(f"Cloud max bound after scale:  {cloud.get_max_bound()}")

cloud = np.asarray(np.concatenate((np.asarray(cloud.points), np.asarray(cloud.colors)), axis=1))

print("Segmenting.")

#   Remove obj and cls id variables.  Input pointcloud will not have GT and we can visualize results.
unequalized_points = cloud
# obj_id = all_obj_id[room_id]
# cls_id = all_cls_id[room_id]

#equalize resolution
t1 = time.time()
equalized_idx = []
unequalized_idx = []
equalized_map = {}
normal_grid = {}
for i in range(len(unequalized_points)):
	k = tuple(np.round(unequalized_points[i,:3]/resolution).astype(int))
	if not k in equalized_map:
		equalized_map[k] = len(equalized_idx)
		equalized_idx.append(i)
	unequalized_idx.append(equalized_map[k])
	if not k in normal_grid:
		normal_grid[k] = []
	normal_grid[k].append(i)
points = unequalized_points[equalized_idx]
# obj_id = obj_id[equalized_idx]
# cls_id = cls_id[equalized_idx]
xyz = points[:,:3]
rgb = points[:,3:6]
room_coordinates = (xyz - xyz.min(axis=0)) / (xyz.max(axis=0) - xyz.min(axis=0))

seed_k = tuple(np.round(seed/resolution).astype(int))
seed_idx = equalized_map[seed_k]

#compute normals
normals = []
curvatures = []
# deletes = []
for i in range(len(points)):
	k = tuple(np.round(points[i,:3]/resolution).astype(int))
	neighbors = []
	for offset in itertools.product([-1,0,1],[-1,0,1],[-1,0,1]):
		kk = (k[0]+offset[0], k[1]+offset[1], k[2]+offset[2])
		if kk in normal_grid:
			neighbors.extend(normal_grid[kk])
	accA = np.zeros((3,3))
	accB = np.zeros(3)
	for n in neighbors:
		p = unequalized_points[n,:3]
		accA += np.outer(p,p)
		accB += p
	cov = accA / len(neighbors) - np.outer(accB, accB) / len(neighbors)**2
	U,S,V = np.linalg.svd(cov)
	curvature = S[2] / (S[0] + S[1] + S[2])

	normals.append(np.fabs(V[2]))
	curvatures.append(np.fabs(curvature))

curvatures = np.array(curvatures)
curvatures = curvatures/np.nanmax(curvatures)
normals = np.array(normals)
if FEATURE_SIZE==6:
	points = np.hstack((xyz, room_coordinates)).astype(np.float32)
elif FEATURE_SIZE==9:
	points = np.hstack((xyz, room_coordinates, rgb)).astype(np.float32)
elif FEATURE_SIZE==12:
	points = np.hstack((xyz, room_coordinates, rgb, normals)).astype(np.float32)
else:
	points = np.hstack((xyz, room_coordinates, rgb, normals, curvatures.reshape(-1,1))).astype(np.float32)

comp_time_analysis['feature'].append(time.time() - t1)

point_voxels = np.round(points[:,:3]/resolution).astype(int)
cluster_label = np.zeros(len(points), dtype=int)
cluster_id = 1
visited = np.zeros(len(point_voxels), dtype=bool)
inlier_points = np.zeros((1, NUM_INLIER_POINT, FEATURE_SIZE), dtype=np.float32)
neighbor_points = np.zeros((1, NUM_NEIGHBOR_POINT, FEATURE_SIZE), dtype=np.float32)
input_add = np.zeros((1, NUM_NEIGHBOR_POINT), dtype=np.int32)
input_remove = np.zeros((1, NUM_INLIER_POINT), dtype=np.int32)
order = np.argsort(curvatures)
#iterate over each object in the room
#		for seed_id in range(len(point_voxels)):
net.eval()
seed_id = seed_idx
seed_voxel = point_voxels[seed_id]

seed_voxel = point_voxels[seed_id]
currentMask = np.zeros(len(points), dtype=bool)
currentMask[seed_id] = True
minDims = seed_voxel.copy()
maxDims = seed_voxel.copy()
seqMinDims = minDims
seqMaxDims = maxDims
steps = 0
stuck = 0
maskLogProb = 0

#perform region growing
while True:

	def stop_growing(reason):
		global cluster_id, start_time
		visited[currentMask] = True
		if np.sum(currentMask) > cluster_threshold:
			cluster_label[currentMask] = cluster_id
			cluster_id += 1

	#determine the current points and the neighboring points
	t = time.time()
	currentPoints = points[currentMask, :].copy()
	newMinDims = minDims.copy()	
	newMaxDims = maxDims.copy()	
	newMinDims -= 1
	newMaxDims += 1
	mask = np.logical_and(np.all(point_voxels>=newMinDims,axis=1), np.all(point_voxels<=newMaxDims, axis=1))
	mask = np.logical_and(mask, np.logical_not(currentMask))
	mask = np.logical_and(mask, np.logical_not(visited))
	expandPoints = points[mask, :].copy()
	
	if len(expandPoints)==0: #no neighbors (early termination)
		stop_growing('noneighbor')
		break

	if len(currentPoints) >= NUM_INLIER_POINT:
		subset = np.random.choice(len(currentPoints), NUM_INLIER_POINT, replace=False)
	else:
		subset = list(range(len(currentPoints))) + list(np.random.choice(len(currentPoints), NUM_INLIER_POINT-len(currentPoints), replace=True))
	center = np.median(currentPoints, axis=0)
	expandPoints = np.array(expandPoints)
	expandPoints[:,:2] -= center[:2]
	expandPoints[:,6:] -= center[6:]
	inlier_points[0,:,:] = currentPoints[subset, :]
	inlier_points[0,:,:2] -= center[:2]
	inlier_points[0,:,6:] -= center[6:]
	if len(expandPoints) >= NUM_NEIGHBOR_POINT:
		subset = np.random.choice(len(expandPoints), NUM_NEIGHBOR_POINT, replace=False)
	else:
		subset = list(range(len(expandPoints))) + list(np.random.choice(len(expandPoints), NUM_NEIGHBOR_POINT-len(expandPoints), replace=True))
	neighbor_points	[0,:,:] = np.array(expandPoints)[subset, :]
	comp_time_analysis['current_neighbor'].append(time.time() - t)
	t = time.time()

	#   Add dummy input_add/remove.
	#   batch=1, seq_length=1.
	inlier_tensor = torch.FloatTensor(inlier_points).to(device)
	inlier_coords = inlier_tensor[:,:,:3].reshape(-1, 3)
	inlier_grid_coords = inlier_tensor[:,:,3:6].reshape(-1, 3)
	inlier_feats = inlier_tensor[:,:,6:FEATURE_SIZE].reshape(-1, FEATURE_SIZE-6)
	inlier_offsets = torch.tensor([(i+1)*NUM_INLIER_POINT for i in range(1)]).to(device)
	inlier_data = {
		'coord': inlier_grid_coords,
		#'grid_coord': inlier_grid_coords,
		'feat': inlier_feats,
		'offset': inlier_offsets,
		'grid_size':  0.01
	}

	neighbor_tensor = torch.FloatTensor(neighbor_points).to(device)
	neighbor_coords = neighbor_tensor[:,:,:3].reshape(-1, 3)
	neighbor_grid_coords = neighbor_tensor[:,:,3:6].reshape(-1, 3)
	neighbor_feats = neighbor_tensor[:,:,6:FEATURE_SIZE].reshape(-1, FEATURE_SIZE-6)
	neighbor_offsets = torch.tensor([(i+1)*NUM_NEIGHBOR_POINT for i in range(1)]).to(device)
	neighbor_data = {
		'coord': neighbor_grid_coords,
		#'grid_coord': neighbor_grid_coords,
		'feat': neighbor_feats,
		'offset': neighbor_offsets,
		'grid_size':  0.01
	}	

	rmv, add = net(inlier_data, neighbor_data)


	comp_time_analysis['current_net'].append(time.time() - t)
	t = time.time()

	add_conf = scipy.special.expit(add.cpu().detach().numpy())
	rmv_conf = scipy.special.expit(rmv.cpu().detach().numpy())
	add_mask = np.random.random(len(add_conf)) < add_conf
	rmv_mask = np.random.random(len(rmv_conf)) < rmv_conf
	addPoints = neighbor_points[0,:,:][add_mask]
	addPoints[:,:2] += center[:2]
	addVoxels = np.round(addPoints[:,:3]/resolution).astype(int)
	addSet = set([tuple(p) for p in addVoxels])
	rmvPoints = inlier_points[0,:,:][rmv_mask]
	rmvPoints[:,:2] += center[:2]
	rmvVoxels = np.round(rmvPoints[:,:3]/resolution).astype(int)
	rmvSet = set([tuple(p) for p in rmvVoxels])
	updated = False
	for i in range(len(point_voxels)):
		if not currentMask[i] and tuple(point_voxels[i]) in addSet:
			currentMask[i] = True
			updated = True
		if tuple(point_voxels[i]) in rmvSet:
			currentMask[i] = False
	steps += 1
	comp_time_analysis['current_inlier'].append(time.time() - t)

	if updated: #continue growing
		minDims = point_voxels[currentMask, :].min(axis=0)
		maxDims = point_voxels[currentMask, :].max(axis=0)
		if not np.any(minDims<seqMinDims) and not np.any(maxDims>seqMaxDims):
			if stuck >= 1:
				stop_growing('stuck')
				break
			else:
				stuck += 1
		else:
			stuck = 0
		seqMinDims = np.minimum(seqMinDims, minDims)
		seqMaxDims = np.maximum(seqMaxDims, maxDims)
	else: #no matching neighbors (early termination)
		stop_growing('noexpand')
		break

#save point cloud results to file
color_sample_state = np.random.RandomState(0)
obj_color = color_sample_state.randint(0,255,(np.max(cluster_label)+1,3))
obj_color[0] = [100,100,100]
obj_color[1] = [255, 0, 0]
unequalized_points[:,3:6] = obj_color[cluster_label,:][unequalized_idx]

#   Propagate segmentation results (colors of output cloud) to source cloud.
result_points = o3d.utility.Vector3dVector(unequalized_points[:, :3])
result_colors = o3d.utility.Vector3dVector(unequalized_points[:, 3:] / 255)
result_cloud = o3d.geometry.PointCloud(result_points)
result_cloud.colors = result_colors

if scale != 1:
	result_cloud = result_cloud.scale(1/scale, result_cloud.get_center())

result_cloud = result_cloud.translate(cloud_center, relative=True)

unique_colors = np.unique(unequalized_points[:, 3:6], axis=0)
print(f"Number of objects detected: {unique_colors.shape[0]}")

o3d.visualization.draw_geometries([result_cloud])

proceed = input("Enter P to proceed to refinement: ")
if proceed.lower() != 'p':
	exit()

print("Segmentation Refinement.  Press Q to exit and save objects.")
result_cloud = utils.merge_segments(result_cloud)

result_points = np.asarray(np.concatenate((np.asarray(result_cloud.points), np.asarray(result_cloud.colors)), axis=1))

unique_colors = np.unique(result_points[:, 3:6], axis=0)
print(f"Number of objects detected: {unique_colors.shape[0]}")

if idx_list is not None:
	print("Begin recoloring.")
	for i in range(len(idx_list)):
		idx = idx_list[i]
		color = result_points[i, 3:]

		source_cloud_np[idx, 3:] = color

	result_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source_cloud_np[:, 0:3]))
	result_cloud.colors = o3d.utility.Vector3dVector(source_cloud_np[:, 3:6])
	result_points = np.asarray(np.concatenate((np.asarray(result_cloud.points), np.asarray(result_cloud.colors)), axis=1))

print("Segmenting and saving objects.")
viz = o3d.visualization.Visualizer()
viz.create_window()

seg_results_path = os.path.join(RESULTS_DIR, f"{os.path.basename(SCAN_NAME)[:-4]}_seg_results_{down_sample}")
os.mkdir(seg_results_path)

#   -90 degree rotation around the x-axis.  FactoryDay3 is initially oriented looking down on the factory (birds-eye view).
#   This rotation orients objects to typical viewing perspective.
rot = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])

#   Extract individual objects for separate file writing.
for i in range(unique_colors.shape[0]):
	object_idx = np.where((result_points[:, 3:6] == unique_colors[i]).all(axis=1))[0]
	object_points = result_points[object_idx, :3]
	object_colors = source_colors[object_idx]

	object_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(object_points))
	object_cloud.colors = o3d.utility.Vector3dVector(object_colors)

	object_cloud = object_cloud.rotate(rot)

	viz.add_geometry(object_cloud)
	viz.capture_screen_image(os.path.join(seg_results_path, f"object_{i}.png"), do_render=True)
	viz.remove_geometry(object_cloud)

	o3d.io.write_point_cloud(os.path.join(seg_results_path, f"object_{i}.pts"), object_cloud)