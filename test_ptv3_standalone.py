import argparse
import itertools
import os
import time

import numpy as np
import torch
from sklearn.cluster import MeanShift
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
)

from learn_region_grow_util import loadFromH5
from ptv3_util import PTv3Segmentation


np.random.seed(0)

RESOLUTION = 0.1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test PTv3 standalone on full point clouds (no region growing)."
    )
    parser.add_argument(
        '--checkpoint', type=str, default='model_ptv3_standalone.pth',
        help="Path to model checkpoint.",
    )
    parser.add_argument(
        '--test_areas', type=str, required=True,
        help="Comma-separated areas to test on (e.g., 'scannet' or '1,2,3,4,5,6').",
    )
    parser.add_argument('--feature_size', type=int, default=12)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument(
        '--bandwidth', type=float, default=1.0,
        help="MeanShift bandwidth for clustering embeddings.",
    )
    parser.add_argument(
        '--save', action='store_true',
        help="Save segmentation results as PLY files.",
    )
    return parser.parse_args()


def prepare_room_features(unequalized_points, resolution, feature_size):
    """Equalize resolution, compute normals, and build feature array for a room."""
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
        if k not in normal_grid:
            normal_grid[k] = []
        normal_grid[k].append(i)

    points = unequalized_points[equalized_idx]
    xyz = points[:, :3]
    rgb = points[:, 3:6]
    room_coordinates = (xyz - xyz.min(axis=0)) / (xyz.max(axis=0) - xyz.min(axis=0))

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
        normals.append(np.fabs(V[2]))
        curvature = S[2] / (S[0] + S[1] + S[2])
        curvatures.append(np.fabs(curvature))

    curvatures = np.array(curvatures)
    curvatures = curvatures / np.nanmax(curvatures)
    normals = np.array(normals)

    if feature_size == 6:
        feat_points = np.hstack((xyz, room_coordinates)).astype(np.float32)
    elif feature_size == 9:
        feat_points = np.hstack((xyz, room_coordinates, rgb)).astype(np.float32)
    elif feature_size == 12:
        feat_points = np.hstack((xyz, room_coordinates, rgb, normals)).astype(np.float32)
    else:
        feat_points = np.hstack(
            (xyz, room_coordinates, rgb, normals, curvatures.reshape(-1, 1))
        ).astype(np.float32)

    return feat_points, equalized_idx, unequalized_idx


def build_ptv3_input(feat_points, feature_size, device):
    coords = torch.FloatTensor(feat_points[:, 3:6]).to(device)
    feats = torch.FloatTensor(feat_points[:, 6:feature_size]).to(device)
    offsets = torch.tensor([len(feat_points)], device=device)
    return {
        'coord': coords,
        'feat': feats,
        'offset': offsets,
        'grid_size': 0.01,
    }


def compute_metrics(obj_id, cluster_label):
    """Compute instance segmentation metrics (same as test_region_grow.py)."""
    nmi = normalized_mutual_info_score(obj_id, cluster_label)
    ami = adjusted_mutual_info_score(obj_id, cluster_label)
    ars = adjusted_rand_score(obj_id, cluster_label)

    # Precision / Recall / IOU via greedy matching (IOU > 0.5)
    gt_match = 0
    dt_match = np.zeros(cluster_label.max(), dtype=bool) if cluster_label.max() > 0 else np.array([], dtype=bool)
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

    prc = np.mean(dt_match) if len(dt_match) > 0 else 0.0
    rcl = 1.0 * gt_match / len(set(obj_id)) if len(set(obj_id)) > 0 else 0.0
    mean_iou = np.mean(room_iou) if room_iou else 0.0

    return nmi, ami, ars, prc, rcl, mean_iou


def main():
    args = parse_args()
    test_areas = [a.strip() for a in args.test_areas.split(',')]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    net = PTv3Segmentation(
        feature_dim=args.feature_size, embed_dim=args.embed_dim
    ).to(device)
    net.eval()

    checkpoint = torch.load(args.checkpoint)
    net.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']} "
          f"(loss {checkpoint['loss']:.4f})")

    agg_nmi, agg_ami, agg_ars = [], [], []
    agg_prc, agg_rcl, agg_iou = [], [], []
    save_id = 0

    for area in test_areas:
        if area in ['scannet', 's3dis', 'kitti_train', 'kitti_val']:
            path = f'data/{area}.h5'
        else:
            path = f'data/s3dis_area{area}.h5'
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue

        all_points, all_obj_id, all_cls_id = loadFromH5(path)
        print(f"\nArea {area}: {len(all_points)} rooms")

        for room_id in range(len(all_points)):
            t_start = time.time()
            unequalized_points = all_points[room_id]
            obj_id = all_obj_id[room_id]

            feat_points, equalized_idx, unequalized_idx = prepare_room_features(
                unequalized_points, RESOLUTION, args.feature_size
            )
            obj_id_eq = obj_id[equalized_idx]

            data = build_ptv3_input(feat_points, args.feature_size, device)

            with torch.no_grad():
                embeddings = net(data).cpu().numpy()

            # Cluster embeddings with MeanShift
            clustering = MeanShift(bandwidth=args.bandwidth, n_jobs=-1)
            cluster_label = clustering.fit_predict(embeddings) + 1  # 1-indexed

            elapsed = time.time() - t_start

            nmi, ami, ars, prc, rcl, mean_iou = compute_metrics(obj_id_eq, cluster_label)
            agg_nmi.append(nmi)
            agg_ami.append(ami)
            agg_ars.append(ars)
            agg_prc.append(prc)
            agg_rcl.append(rcl)
            agg_iou.append(mean_iou)

            n_clusters = cluster_label.max()
            n_gt = len(set(obj_id_eq))
            print(
                f"Area {area} room {room_id:3d} ({len(feat_points):6d} pts, "
                f"{n_clusters}/{n_gt} clusters) "
                f"NMI={nmi:.2f} AMI={ami:.2f} ARS={ars:.2f} "
                f"PRC={prc:.2f} RCL={rcl:.2f} IOU={mean_iou:.2f} "
                f"({elapsed:.1f}s)"
            )

            if args.save:
                from learn_region_grow_util import savePLY
                color_state = np.random.RandomState(0)
                obj_color = color_state.randint(0, 255, (n_clusters + 1, 3))
                obj_color[0] = [100, 100, 100]
                uneq_labels = cluster_label[unequalized_idx]
                uneq_pts = unequalized_points.copy()
                uneq_pts[:, 3:6] = obj_color[uneq_labels, :]
                savePLY(f'data/results/ptv3_standalone/{save_id}.ply', uneq_pts)
                save_id += 1

    if agg_nmi:
        print(
            f"\nOverall ({len(agg_nmi)} rooms): "
            f"NMI={np.mean(agg_nmi):.2f}+-{np.std(agg_nmi):.2f} "
            f"AMI={np.mean(agg_ami):.2f}+-{np.std(agg_ami):.2f} "
            f"ARS={np.mean(agg_ars):.2f}+-{np.std(agg_ars):.2f} "
            f"PRC={np.mean(agg_prc):.2f}+-{np.std(agg_prc):.2f} "
            f"RCL={np.mean(agg_rcl):.2f}+-{np.std(agg_rcl):.2f} "
            f"IOU={np.mean(agg_iou):.2f}+-{np.std(agg_iou):.2f}"
        )


if __name__ == '__main__':
    main()
