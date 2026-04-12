import argparse
import itertools
import os
import time
from collections import defaultdict

import numpy as np
import scipy.special
import torch

from class_util import classes_s3dis, classes_nyu40, classes_kitti
from learn_region_grow_util import loadFromH5
from ptv3_util import RegionTransformerPTv3


# Helpers below are copied verbatim from test_region_grow_single_instance.py.
# They are inlined rather than imported because that script imports open3d at
# module top-level, which is not needed for this headless benchmark.
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


def build_ptv3_input(points_tensor, num_points, feature_size, device):
    grid_coords = points_tensor[:, :, 3:6].reshape(-1, 3)
    feats = points_tensor[:, :, 6:feature_size].reshape(-1, feature_size - 6)
    offsets = torch.tensor([num_points], device=device)
    return {
        'coord': grid_coords,
        'feat': feats,
        'offset': offsets,
        'grid_size': 0.01,
    }


np.random.seed(0)

FEATURE_SIZE = 12
RESOLUTION = 0.1
CLUSTER_THRESHOLD = 10
CHECKPOINT_PATH = "model_checkpoint.pth"
TEST_AREAS = ['1', '2', '3', '4', '5', '6', 'scannet']


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_points', type=int, default=512)
    parser.add_argument(
        '--areas',
        type=str,
        default=','.join(TEST_AREAS),
        help="Comma-separated list of areas to benchmark (default: all TEST_AREAS).",
    )
    parser.add_argument(
        '--max_rooms_per_area',
        type=int,
        default=0,
        help="Cap rooms per area (0 = unlimited).",
    )
    args = parser.parse_args()
    args.areas = [a.strip() for a in args.areas.split(',') if a.strip()]
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
    net, points, point_voxels, seed_id, visited, num_points, feature_size, device
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

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()

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

    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - start
    return current_mask, elapsed, steps


def print_summary(label, times, iterations):
    if not times:
        print(f"  {label}: no samples")
        return
    t = np.array(times)
    it = np.array(iterations)
    print(
        f"  {label}: count={len(t)}  "
        f"time  min={t.min():.4f}s  max={t.max():.4f}s  mean={t.mean():.4f}s  median={np.median(t):.4f}s  "
        f"iters  min={it.min()}  max={it.max()}  mean={it.mean():.1f}  median={np.median(it):.1f}"
    )


def print_class_table(label, class_stats):
    """Print a summary table for all classes in class_stats."""
    if not class_stats:
        return
    print(f"\n{label}:")
    for cls_name in sorted(class_stats, key=lambda c: len(class_stats[c]['times']), reverse=True):
        entry = class_stats[cls_name]
        print_summary(cls_name, entry['times'], entry['iters'])


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = RegionTransformerPTv3(
        1, 1, args.num_points, args.num_points, FEATURE_SIZE
    ).to(device)
    net.eval()
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])

    overall_stats = defaultdict(lambda: {'times': [], 'iters': []})

    for area in args.areas:
        print(f"\n=== Area {area} ===")
        all_points, all_obj_id, all_cls_id, room_names, sample_list = load_area(area)
        classes = classes_kitti if 'kitti' in area else classes_nyu40 if area == 'scannet' else classes_s3dis
        area_stats = defaultdict(lambda: {'times': [], 'iters': []})

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
            obj_id = all_obj_id[room_id]
            cls_id = all_cls_id[room_id]
            points, point_voxels, curvatures, equalized_idx = prepare_room_features(
                unequalized_points
            )
            obj_id = obj_id[equalized_idx]
            cls_id = cls_id[equalized_idx]
            visited = np.zeros(len(points), dtype=bool)
            order = np.argsort(curvatures)

            room_times = []
            for seed_id in order:
                if visited[seed_id]:
                    continue
                cls_name = classes[cls_id[seed_id]]
                _, elapsed, steps = grow_single_seed(
                    net,
                    points,
                    point_voxels,
                    int(seed_id),
                    visited,
                    args.num_points,
                    FEATURE_SIZE,
                    device,
                )
                area_stats[cls_name]['times'].append(elapsed)
                area_stats[cls_name]['iters'].append(steps)
                overall_stats[cls_name]['times'].append(elapsed)
                overall_stats[cls_name]['iters'].append(steps)
                room_times.append(elapsed)

            if room_times:
                rt = np.array(room_times)
                print(
                    f"area {area} room {room_id:4d}: "
                    f"{len(rt):4d} instances  "
                    f"min {rt.min():.4f}s  max {rt.max():.4f}s  "
                    f"mean {rt.mean():.4f}s  median {np.median(rt):.4f}s"
                )
            else:
                print(f"area {area} room {room_id:4d}: no instances grown")

        print_class_table(f"Area {area} per-class summary", area_stats)

    print_class_table("Overall per-class summary", overall_stats)

    # Overall totals across all classes
    all_times = []
    all_iters = []
    for entry in overall_stats.values():
        all_times.extend(entry['times'])
        all_iters.extend(entry['iters'])
    if all_times:
        print(f"\nOverall totals:")
        print_summary("all classes", all_times, all_iters)


if __name__ == '__main__':
    main()
