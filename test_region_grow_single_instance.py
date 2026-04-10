import argparse
import itertools
import time

import numpy as np
import open3d as o3d
import scipy.special
import torch

import pointcloud_utils as utils
from ptv3_util import RegionTransformerPTv3


np.random.seed(0)

# Curvatures may contain nan values. Using 12 features instead of 13.
FEATURE_SIZE = 12
RESOLUTION = 0.1
CLUSTER_THRESHOLD = 10
SCAN_NAME = "factory_data/Factory-small-downsampled.ply"
CHECKPOINT_PATH = "model_checkpoint.pth"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_points', type=int, default=512)
    parser.add_argument('--downsample', type=float, default=0)
    parser.add_argument('--scale', type=float, default=1)
    args = parser.parse_args()
    if args.scale == 0:
        raise SystemExit("Scale cannot be 0 as this entails zero division.")
    return args


def load_cloud(scan_name, downsample, scale):
    print("Loading pointcloud.")
    source_cloud = o3d.io.read_point_cloud(scan_name)

    seed_idx = utils.pick_points(source_cloud)
    seeds = np.asarray(source_cloud.points)[seed_idx, :]
    print(seeds)
    print(f"Cloud point count:  {np.asarray(source_cloud.points).shape[0]}")

    if downsample != 0:
        print("Downsampling.")
        cloud, _, _ = source_cloud.voxel_down_sample_and_trace(
            downsample, source_cloud.get_min_bound(), source_cloud.get_max_bound()
        )
        print(f"Cloud point count after downsample:  {np.asarray(cloud.points).shape[0]}")
    else:
        cloud = source_cloud

    cloud_center = cloud.get_center()
    cloud = cloud.translate(-1 * cloud_center, relative=True)
    seeds = seeds - cloud_center

    print(f"Cloud min bound:  {cloud.get_min_bound()}")
    print(f"Cloud max bound:  {cloud.get_max_bound()}")

    if scale != 1:
        cloud = cloud.scale(scale, cloud.get_center())
        print(f"Cloud min bound after scale:  {cloud.get_min_bound()}")
        print(f"Cloud max bound after scale:  {cloud.get_max_bound()}")

    cloud_np = np.concatenate(
        (np.asarray(cloud.points), np.asarray(cloud.colors)), axis=1
    )
    return cloud_np, cloud_center, seeds


def equalize_resolution(unequalized_points, resolution):
    equalized_idx = []
    unequalized_idx = []
    equalized_map = {}
    normal_grid = {}
    for i in range(len(unequalized_points)):
        k = tuple(np.round(unequalized_points[i, :3] / resolution).astype(int))
        if k not in equalized_map:
            equalized_map[k] = len(equalized_idx)
            equalized_idx.append(i)
        unequalized_idx.append(equalized_map[k])
        normal_grid.setdefault(k, []).append(i)
    return equalized_idx, unequalized_idx, equalized_map, normal_grid


def compute_normals(points, unequalized_points, normal_grid, resolution):
    normals = []
    curvatures = []
    for i in range(len(points)):
        k = tuple(np.round(points[i, :3] / resolution).astype(int))
        neighbors = []
        for offset in itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]):
            kk = (k[0] + offset[0], k[1] + offset[1], k[2] + offset[2])
            if kk in normal_grid:
                neighbors.extend(normal_grid[kk])
        accA = np.zeros((3, 3))
        accB = np.zeros(3)
        for n in neighbors:
            p = unequalized_points[n, :3]
            accA += np.outer(p, p)
            accB += p
        cov = accA / len(neighbors) - np.outer(accB, accB) / len(neighbors) ** 2
        _, S, V = np.linalg.svd(cov)
        curvature = S[2] / (S[0] + S[1] + S[2])
        normals.append(np.fabs(V[2]))
        curvatures.append(np.fabs(curvature))

    curvatures = np.array(curvatures)
    curvatures = curvatures / np.nanmax(curvatures)
    normals = np.array(normals)
    return normals, curvatures


def build_feature_points(xyz, rgb, room_coordinates, normals, curvatures, feature_size):
    if feature_size == 6:
        return np.hstack((xyz, room_coordinates)).astype(np.float32)
    if feature_size == 9:
        return np.hstack((xyz, room_coordinates, rgb)).astype(np.float32)
    if feature_size == 12:
        return np.hstack((xyz, room_coordinates, rgb, normals)).astype(np.float32)
    return np.hstack(
        (xyz, room_coordinates, rgb, normals, curvatures.reshape(-1, 1))
    ).astype(np.float32)


def prepare_features(unequalized_points, resolution, feature_size):
    equalized_idx, unequalized_idx, equalized_map, normal_grid = equalize_resolution(
        unequalized_points, resolution
    )
    points = unequalized_points[equalized_idx]
    xyz = points[:, :3]
    rgb = points[:, 3:6]
    room_coordinates = (xyz - xyz.min(axis=0)) / (xyz.max(axis=0) - xyz.min(axis=0))

    normals, curvatures = compute_normals(
        points, unequalized_points, normal_grid, resolution
    )
    feature_points = build_feature_points(
        xyz, rgb, room_coordinates, normals, curvatures, feature_size
    )
    return feature_points, unequalized_idx, equalized_map


def sample_indices(count, target):
    if count >= target:
        return np.random.choice(count, target, replace=False)
    return list(range(count)) + list(
        np.random.choice(count, target - count, replace=True)
    )


def build_ptv3_input(points_tensor, num_points, feature_size, device):
    coords = points_tensor[:, :, :3].reshape(-1, 3)
    grid_coords = points_tensor[:, :, 3:6].reshape(-1, 3)
    feats = points_tensor[:, :, 6:feature_size].reshape(-1, feature_size - 6)
    offsets = torch.tensor([num_points], device=device)
    return {
        'coord': grid_coords,
        'feat': feats,
        'offset': offsets,
        'grid_size': 0.01,
    }


def run_region_growing(net, points, point_voxels, seed_idxs, num_points, feature_size, device):
    cluster_label = np.zeros(len(points), dtype=int)
    visited = np.zeros(len(point_voxels), dtype=bool)
    inlier_points = np.zeros((1, num_points, feature_size), dtype=np.float32)
    neighbor_points = np.zeros((1, num_points, feature_size), dtype=np.float32)

    seed_voxel = point_voxels[seed_idxs[0]]
    current_mask = np.zeros(len(points), dtype=bool)
    for seed in seed_idxs:
        current_mask[seed] = True

    min_dims = seed_voxel.copy()
    max_dims = seed_voxel.copy()
    seq_min_dims = min_dims
    seq_max_dims = max_dims
    stuck = 0

    def stop_growing():
        visited[current_mask] = True
        if np.sum(current_mask) > CLUSTER_THRESHOLD:
            cluster_label[current_mask] = 1

    start_time = time.time()

    while True:
        current_points = points[current_mask, :].copy()
        new_min_dims = min_dims.copy() - 1
        new_max_dims = max_dims.copy() + 1
        mask = np.logical_and(
            np.all(point_voxels >= new_min_dims, axis=1),
            np.all(point_voxels <= new_max_dims, axis=1),
        )
        mask = np.logical_and(mask, np.logical_not(current_mask))
        mask = np.logical_and(mask, np.logical_not(visited))
        expand_points = points[mask, :].copy()

        if len(expand_points) == 0:
            stop_growing()
            break

        inlier_subset = sample_indices(len(current_points), num_points)
        center = np.median(current_points, axis=0)
        expand_points[:, :2] -= center[:2]
        expand_points[:, 6:] -= center[6:]
        inlier_points[0, :, :] = current_points[inlier_subset, :]
        inlier_points[0, :, :2] -= center[:2]
        inlier_points[0, :, 6:] -= center[6:]

        neighbor_subset = sample_indices(len(expand_points), num_points)
        neighbor_points[0, :, :] = expand_points[neighbor_subset, :]

        inlier_tensor = torch.FloatTensor(inlier_points).to(device)
        neighbor_tensor = torch.FloatTensor(neighbor_points).to(device)
        inlier_data = build_ptv3_input(inlier_tensor, num_points, feature_size, device)
        neighbor_data = build_ptv3_input(neighbor_tensor, num_points, feature_size, device)

        rmv, add = net(inlier_data, neighbor_data)

        add_conf = scipy.special.expit(add.cpu().detach().numpy())
        rmv_conf = scipy.special.expit(rmv.cpu().detach().numpy())
        add_mask = np.random.random(len(add_conf)) < add_conf
        rmv_mask = np.random.random(len(rmv_conf)) < rmv_conf

        add_pts = neighbor_points[0, :, :][add_mask]
        add_pts[:, :2] += center[:2]
        add_voxels = np.round(add_pts[:, :3] / RESOLUTION).astype(int)
        add_set = set(tuple(p) for p in add_voxels)

        rmv_pts = inlier_points[0, :, :][rmv_mask]
        rmv_pts[:, :2] += center[:2]
        rmv_voxels = np.round(rmv_pts[:, :3] / RESOLUTION).astype(int)
        rmv_set = set(tuple(p) for p in rmv_voxels)

        updated = False
        for i in range(len(point_voxels)):
            vox = tuple(point_voxels[i])
            if not current_mask[i] and vox in add_set:
                current_mask[i] = True
                updated = True
            if vox in rmv_set:
                current_mask[i] = False

        for seed in seed_idxs:
            if not current_mask[seed]:
                current_mask[seed] = True
                stop_growing()
                break

        if updated:
            min_dims = point_voxels[current_mask, :].min(axis=0)
            max_dims = point_voxels[current_mask, :].max(axis=0)
            if not np.any(min_dims < seq_min_dims) and not np.any(max_dims > seq_max_dims):
                if stuck >= 1:
                    stop_growing()
                    break
                stuck += 1
            else:
                stuck = 0
            seq_min_dims = np.minimum(seq_min_dims, min_dims)
            seq_max_dims = np.maximum(seq_max_dims, max_dims)
        else:
            stop_growing()
            break

    print(f"Region growing completed in {time.time() - start_time} seconds.")
    if not current_mask[seed_idxs[0]]:
        print("Seed point was removed during region growing.")
    return cluster_label


def build_result_cloud(unequalized_points, cluster_label, unequalized_idx, cloud_center, scale):
    color_sample_state = np.random.RandomState(0)
    obj_color = color_sample_state.randint(0, 255, (np.max(cluster_label) + 1, 3))
    obj_color[0] = [100, 100, 100]
    obj_color[1] = [255, 0, 0]
    unequalized_points[:, 3:6] = obj_color[cluster_label, :][unequalized_idx]

    result_cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(unequalized_points[:, :3])
    )
    result_cloud.colors = o3d.utility.Vector3dVector(unequalized_points[:, 3:] / 255)

    if scale != 1:
        result_cloud = result_cloud.scale(1 / scale, result_cloud.get_center())
    result_cloud = result_cloud.translate(cloud_center, relative=True)

    unique_colors = np.unique(unequalized_points[:, 3:6], axis=0)
    return result_cloud, unique_colors


def main():
    args = parse_args()
    num_points = args.num_points

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = RegionTransformerPTv3(1, 1, num_points, num_points, FEATURE_SIZE).to(device)
    net.eval()
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])

    cloud_np, cloud_center, seeds = load_cloud(
        SCAN_NAME, args.downsample, args.scale
    )

    print("Segmenting.")
    unequalized_points = cloud_np
    points, unequalized_idx, equalized_map = prepare_features(
        unequalized_points, RESOLUTION, FEATURE_SIZE
    )
    point_voxels = np.round(points[:, :3] / RESOLUTION).astype(int)
    seed_idxs = [
        equalized_map[tuple(np.round(seed / RESOLUTION).astype(int))] for seed in seeds
    ]

    cluster_label = run_region_growing(
        net, points, point_voxels, seed_idxs, num_points, FEATURE_SIZE, device
    )

    result_cloud, unique_colors = build_result_cloud(
        unequalized_points, cluster_label, unequalized_idx, cloud_center, args.scale
    )
    print(f"Number of objects detected: {unique_colors.shape[0]}")
    o3d.visualization.draw_geometries([result_cloud])


if __name__ == '__main__':
    main()
