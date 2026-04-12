import argparse
import itertools
import os
import time

import numpy as np
import scipy.special
import torch
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from learn_region_grow_util import loadFromH5
from ptv3_util import RegionTransformerPTv3


# Helpers inlined to avoid open3d dependency from other scripts.
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


def sample_indices(count, target):
    if count >= target:
        return np.random.choice(count, target, replace=False)
    return list(range(count)) + list(
        np.random.choice(count, target - count, replace=True)
    )


def build_ptv3_input(points_tensor, num_points, feature_size, device, grid_size):
    grid_coords = points_tensor[:, :, 3:6].reshape(-1, 3)
    feats = points_tensor[:, :, 6:feature_size].reshape(-1, feature_size - 6)
    offsets = torch.tensor([num_points], device=device)
    return {
        'coord': grid_coords,
        'feat': feats,
        'offset': offsets,
        'grid_size': grid_size,
    }


np.random.seed(0)

FEATURE_SIZE = 12
RESOLUTION = 0.1
CLUSTER_THRESHOLD = 10
CHECKPOINT_PATH = "model_checkpoint.pth"
TEST_AREAS = ['1', '2', '3', '4', '5', '6', 'scannet']


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test effect of grid_size on segmentation quality."
    )
    parser.add_argument('--num_points', type=int, default=512)
    parser.add_argument(
        '--areas',
        type=str,
        default=','.join(TEST_AREAS),
        help="Comma-separated list of areas (default: all).",
    )
    parser.add_argument(
        '--max_rooms_per_area',
        type=int,
        default=0,
        help="Cap rooms per area (0 = unlimited).",
    )
    parser.add_argument(
        '--grid_sizes',
        type=str,
        default='0.005,0.01,0.02,0.05',
        help="Comma-separated grid_size values to test.",
    )
    args = parser.parse_args()
    args.areas = [a.strip() for a in args.areas.split(',') if a.strip()]
    args.grid_sizes = [float(g.strip()) for g in args.grid_sizes.split(',') if g.strip()]
    return args


def load_area(area):
    path = "data/scannet.h5" if area == 'scannet' else f"data/s3dis_area{area}.h5"
    all_points, all_obj_id, all_cls_id = loadFromH5(path)

    room_name_file = (
        "data/scannet_room_name.txt" if area == 'scannet' else "data/s3dis_room_name.txt"
    )
    if os.path.exists(room_name_file):
        room_names = open(room_name_file).read().split('\n')
    else:
        room_names = None

    sample_list = set(open('data/s3dis_sampled.txt').read().split('\n'))
    return all_points, all_obj_id, all_cls_id, room_names, sample_list


def prepare_room_features(unequalized_points):
    equalized_idx, _, _, normal_grid = equalize_resolution(unequalized_points, RESOLUTION)
    eq = unequalized_points[equalized_idx]
    xyz = eq[:, :3]
    rgb = eq[:, 3:6]
    room_coords = (xyz - xyz.min(axis=0)) / (xyz.max(axis=0) - xyz.min(axis=0))
    normals, curvatures = compute_normals(eq, unequalized_points, normal_grid, RESOLUTION)
    points = build_feature_points(
        xyz, rgb, room_coords, normals, curvatures, FEATURE_SIZE
    )
    point_voxels = np.round(points[:, :3] / RESOLUTION).astype(int)
    return points, point_voxels, curvatures, equalized_idx


def grow_single_seed(
    net, points, point_voxels, seed_id, visited, num_points, feature_size, device,
    grid_size,
):
    inlier_points = np.zeros((1, num_points, feature_size), dtype=np.float32)
    neighbor_points = np.zeros((1, num_points, feature_size), dtype=np.float32)

    seed_voxel = point_voxels[seed_id]
    current_mask = np.zeros(len(points), dtype=bool)
    current_mask[seed_id] = True

    min_dims = seed_voxel.copy()
    max_dims = seed_voxel.copy()
    seq_min_dims = min_dims
    seq_max_dims = max_dims
    stuck = 0
    steps = 0

    def stop_growing():
        visited[current_mask] = True

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
        inlier_data = build_ptv3_input(
            inlier_tensor, num_points, feature_size, device, grid_size
        )
        neighbor_data = build_ptv3_input(
            neighbor_tensor, num_points, feature_size, device, grid_size
        )

        rmv, add = net(inlier_data, neighbor_data, grid_size=grid_size)

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

        steps += 1

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

    return current_mask, steps


def evaluate_room(cluster_label, obj_id):
    """Compute segmentation quality metrics for a single room."""
    # Fill unlabeled points by nearest labeled neighbor
    nonzero_idx = np.nonzero(cluster_label)[0]
    if len(nonzero_idx) == 0:
        return None
    nonzero_points_labels = cluster_label[nonzero_idx]
    filled = cluster_label.copy()
    for i in np.nonzero(cluster_label == 0)[0]:
        # Use index distance as proxy (same as test_region_grow.py uses point distance,
        # but we don't have the points array here — caller passes obj_id aligned to points)
        filled[i] = cluster_label[nonzero_idx[0]]
    cluster_label = filled

    # Match predicted clusters to GT objects
    gt_match = 0
    dt_match = np.zeros(cluster_label.max(), dtype=bool)
    room_iou = []
    unique_id, count = np.unique(obj_id, return_counts=True)
    for k in range(len(unique_id)):
        i = unique_id[np.argsort(count)][::-1][k]
        best_iou = 0
        for j in range(1, cluster_label.max() + 1):
            if not dt_match[j - 1]:
                iou = (
                    1.0
                    * np.sum(np.logical_and(obj_id == i, cluster_label == j))
                    / np.sum(np.logical_or(obj_id == i, cluster_label == j))
                )
                best_iou = max(best_iou, iou)
                if iou > 0.5:
                    dt_match[j - 1] = True
                    gt_match += 1
                    break
        room_iou.append(best_iou)

    prc = np.mean(dt_match)
    rcl = 1.0 * gt_match / len(set(obj_id))
    mean_iou = np.mean(room_iou)
    nmi = normalized_mutual_info_score(obj_id, cluster_label)
    ami = adjusted_mutual_info_score(obj_id, cluster_label)
    ars = adjusted_rand_score(obj_id, cluster_label)
    return nmi, ami, ars, prc, rcl, mean_iou


def run_grid_size(net, args, grid_size, device):
    """Run full evaluation for a single grid_size value. Returns aggregated metrics."""
    agg = {'nmi': [], 'ami': [], 'ars': [], 'prc': [], 'rcl': [], 'iou': []}

    for area in args.areas:
        all_points, all_obj_id, all_cls_id, room_names, sample_list = load_area(area)

        rooms_done = 0
        for room_id in range(len(all_points)):
            if room_names is not None:
                base = '_'.join(room_names[room_id].split())
                if f"{base}.npy" not in sample_list and f"{base}.h5" not in sample_list:
                    continue
            if args.max_rooms_per_area and rooms_done >= args.max_rooms_per_area:
                break
            rooms_done += 1

            unequalized_points = all_points[room_id]
            obj_id_raw = all_obj_id[room_id]
            points, point_voxels, curvatures, equalized_idx = prepare_room_features(
                unequalized_points
            )
            obj_id = obj_id_raw[equalized_idx]
            visited = np.zeros(len(points), dtype=bool)
            order = np.argsort(curvatures)

            cluster_label = np.zeros(len(points), dtype=int)
            cluster_id = 1

            for seed_id in order:
                if visited[seed_id]:
                    continue
                current_mask, steps = grow_single_seed(
                    net,
                    points,
                    point_voxels,
                    int(seed_id),
                    visited,
                    args.num_points,
                    FEATURE_SIZE,
                    device,
                    grid_size,
                )
                if np.sum(current_mask) > CLUSTER_THRESHOLD:
                    cluster_label[current_mask] = cluster_id
                    cluster_id += 1

            # Fill unlabeled points by nearest labeled neighbor
            nonzero_idx = np.nonzero(cluster_label)[0]
            if len(nonzero_idx) > 0:
                nonzero_points = points[nonzero_idx, :]
                filled = cluster_label.copy()
                for i in np.nonzero(cluster_label == 0)[0]:
                    d = np.sum((nonzero_points - points[i]) ** 2, axis=1)
                    filled[i] = cluster_label[nonzero_idx[np.argmin(d)]]
                cluster_label = filled

            result = evaluate_room(cluster_label, obj_id)
            if result is None:
                print(f"  area {area} room {room_id:4d}: no clusters")
                continue
            nmi, ami, ars, prc, rcl, mean_iou = result
            agg['nmi'].append(nmi)
            agg['ami'].append(ami)
            agg['ars'].append(ars)
            agg['prc'].append(prc)
            agg['rcl'].append(rcl)
            agg['iou'].append(mean_iou)
            print(
                f"  area {area} room {room_id:4d}: "
                f"NMI={nmi:.3f} AMI={ami:.3f} ARS={ars:.3f} "
                f"PRC={prc:.3f} RCL={rcl:.3f} IOU={mean_iou:.3f}"
            )

    return agg


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = RegionTransformerPTv3(
        1, 1, args.num_points, args.num_points, FEATURE_SIZE
    ).to(device)
    net.eval()
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])

    all_results = {}

    for grid_size in args.grid_sizes:
        print(f"\n{'=' * 60}")
        print(f"grid_size = {grid_size}")
        print('=' * 60)
        np.random.seed(0)
        t0 = time.time()
        agg = run_grid_size(net, args, grid_size, device)
        elapsed = time.time() - t0
        all_results[grid_size] = agg
        if agg['nmi']:
            print(
                f"\n  grid_size={grid_size} summary ({len(agg['nmi'])} rooms, {elapsed:.1f}s): "
                f"NMI={np.mean(agg['nmi']):.3f}+-{np.std(agg['nmi']):.3f} "
                f"AMI={np.mean(agg['ami']):.3f}+-{np.std(agg['ami']):.3f} "
                f"ARS={np.mean(agg['ars']):.3f}+-{np.std(agg['ars']):.3f} "
                f"PRC={np.mean(agg['prc']):.3f}+-{np.std(agg['prc']):.3f} "
                f"RCL={np.mean(agg['rcl']):.3f}+-{np.std(agg['rcl']):.3f} "
                f"IOU={np.mean(agg['iou']):.3f}+-{np.std(agg['iou']):.3f}"
            )

    # Final comparison table
    print(f"\n{'=' * 60}")
    print("Summary comparison across grid_size values")
    print('=' * 60)
    header = f"{'grid_size':>10s}  {'rooms':>5s}  {'NMI':>12s}  {'AMI':>12s}  {'ARS':>12s}  {'PRC':>12s}  {'RCL':>12s}  {'IOU':>12s}"
    print(header)
    print('-' * len(header))
    for gs in args.grid_sizes:
        agg = all_results[gs]
        if not agg['nmi']:
            print(f"{gs:>10.4f}  {'0':>5s}  {'N/A':>12s}  {'N/A':>12s}  {'N/A':>12s}  {'N/A':>12s}  {'N/A':>12s}  {'N/A':>12s}")
            continue
        def fmt(vals):
            return f"{np.mean(vals):.3f}+-{np.std(vals):.3f}"
        print(
            f"{gs:>10.4f}  {len(agg['nmi']):>5d}  "
            f"{fmt(agg['nmi']):>12s}  {fmt(agg['ami']):>12s}  {fmt(agg['ars']):>12s}  "
            f"{fmt(agg['prc']):>12s}  {fmt(agg['rcl']):>12s}  {fmt(agg['iou']):>12s}"
        )


if __name__ == '__main__':
    main()
