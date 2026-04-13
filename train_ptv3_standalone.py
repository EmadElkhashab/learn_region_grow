import argparse
import itertools
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from learn_region_grow_util import loadFromH5
from ptv3_util import PTv3Segmentation, save_checkpoint


np.random.seed(0)
np.set_printoptions(2, linewidth=100, suppress=True, sign=' ')

RESOLUTION = 0.1


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PTv3 standalone on full point clouds for instance segmentation."
    )
    parser.add_argument(
        '--train_areas', type=str, required=True,
        help="Comma-separated areas to train on (e.g., '1,2,3,4,6' or 'scannet').",
    )
    parser.add_argument(
        '--val_areas', type=str, default='',
        help="Comma-separated areas for validation (e.g., '5'). Empty = no validation.",
    )
    parser.add_argument(
        '--checkpoint', type=str, default='model_ptv3_standalone.pth',
        help="Path to save/resume checkpoint.",
    )
    parser.add_argument('--max_epochs', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--feature_size', type=int, default=12)
    parser.add_argument('--embed_dim', type=int, default=16)
    parser.add_argument(
        '--val_step', type=int, default=5,
        help="Run validation every N epochs.",
    )
    parser.add_argument(
        '--max_points', type=int, default=80000,
        help="Max points per room after equalization. Subsample if larger.",
    )
    return parser.parse_args()


def prepare_room_features(unequalized_points, resolution, feature_size):
    """Equalize resolution, compute normals, and build feature array for a room.

    Same pipeline as test_region_grow.py lines 101-153.
    """
    # Equalize resolution
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

    # Compute normals
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

    return feat_points, equalized_idx


def load_rooms(areas, feature_size):
    """Load and prepare all rooms from the given areas."""
    all_rooms = []
    all_labels = []

    for area in areas:
        if area in ['scannet', 's3dis', 'kitti_train', 'kitti_val']:
            path = f'data/{area}.h5'
        else:
            path = f'data/s3dis_area{area}.h5'
        if not os.path.exists(path):
            print(f"Warning: {path} not found, skipping.")
            continue

        room_points, room_obj_id, _ = loadFromH5(path)
        print(f"Loaded {path}: {len(room_points)} rooms")

        for room_id in range(len(room_points)):
            feat_points, equalized_idx = prepare_room_features(
                room_points[room_id], RESOLUTION, feature_size
            )
            obj_id = room_obj_id[room_id][equalized_idx]
            all_rooms.append(feat_points)
            all_labels.append(obj_id)

    return all_rooms, all_labels


class DiscriminativeLoss(nn.Module):
    """Discriminative loss for instance segmentation embeddings.

    From: "Semantic Instance Segmentation with a Discriminative Loss Function"
    (De Brabandere et al., 2017)
    """

    def __init__(self, delta_var=0.5, delta_dist=1.5, alpha=1.0, beta=1.0, gamma=0.001):
        super().__init__()
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, embeddings, labels):
        """
        embeddings: (N, embed_dim) per-point embeddings
        labels: (N,) instance IDs
        """
        unique_labels = torch.unique(labels)
        num_instances = len(unique_labels)

        if num_instances <= 1:
            return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

        # Compute instance means
        means = []
        for lbl in unique_labels:
            mask = labels == lbl
            means.append(embeddings[mask].mean(dim=0))
        means = torch.stack(means)  # (C, embed_dim)

        # Variance (pull) loss: pull points toward their instance mean
        var_loss = torch.tensor(0.0, device=embeddings.device)
        for i, lbl in enumerate(unique_labels):
            mask = labels == lbl
            inst_embed = embeddings[mask]  # (Nc, embed_dim)
            dist = torch.norm(inst_embed - means[i].unsqueeze(0), dim=1)
            dist = torch.clamp(dist - self.delta_var, min=0) ** 2
            var_loss += dist.mean()
        var_loss /= num_instances

        # Distance (push) loss: push instance means apart
        dist_loss = torch.tensor(0.0, device=embeddings.device)
        count = 0
        for i in range(num_instances):
            for j in range(i + 1, num_instances):
                dist = torch.norm(means[i] - means[j])
                dist = torch.clamp(2 * self.delta_dist - dist, min=0) ** 2
                dist_loss += dist
                count += 1
        if count > 0:
            dist_loss /= count

        # Regularization loss
        reg_loss = torch.norm(means, dim=1).mean()

        return self.alpha * var_loss + self.beta * dist_loss + self.gamma * reg_loss


def build_ptv3_input(feat_points, feature_size, device):
    """Build PTv3 input dict from a single room's feature array."""
    coords = torch.FloatTensor(feat_points[:, 3:6]).to(device)
    feats = torch.FloatTensor(feat_points[:, 6:feature_size]).to(device)
    offsets = torch.tensor([len(feat_points)], device=device)
    return {
        'coord': coords,
        'feat': feats,
        'offset': offsets,
        'grid_size': 0.01,
    }


def main():
    args = parse_args()
    train_areas = [a.strip() for a in args.train_areas.split(',')]
    val_areas = [a.strip() for a in args.val_areas.split(',') if a.strip()]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    net = PTv3Segmentation(
        feature_dim=args.feature_size, embed_dim=args.embed_dim
    ).to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr)
    criterion = DiscriminativeLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64, eta_min=0)

    start_epoch = 0
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint['epoch']} with loss {checkpoint['loss']:.4f}")
        for _ in range(start_epoch):
            scheduler.step()

    # Load and prepare rooms
    print(f"\nPreparing training rooms: {train_areas}")
    train_rooms, train_labels = load_rooms(train_areas, args.feature_size)
    print(f"Training rooms: {len(train_rooms)}")

    val_rooms, val_labels = [], []
    if val_areas:
        print(f"\nPreparing validation rooms: {val_areas}")
        val_rooms, val_labels = load_rooms(val_areas, args.feature_size)
        print(f"Validation rooms: {len(val_rooms)}")

    if len(train_rooms) == 0:
        raise SystemExit("No training data found.")

    epoch_times = []

    for epoch in range(start_epoch, args.max_epochs):
        # Training
        idx = np.arange(len(train_rooms))
        np.random.shuffle(idx)
        loss_arr = []
        start_time = time.time()
        net.train()

        for i in tqdm(idx, desc=f"Epoch {epoch + 1} train"):
            room_feats = train_rooms[i]
            room_labels = train_labels[i]

            # Subsample if room is too large
            if len(room_feats) > args.max_points:
                subset = np.random.choice(len(room_feats), args.max_points, replace=False)
                room_feats = room_feats[subset]
                room_labels = room_labels[subset]

            data = build_ptv3_input(room_feats, args.feature_size, device)
            labels_tensor = torch.LongTensor(room_labels).to(device)

            optimizer.zero_grad()
            embeddings = net(data)
            loss = criterion(embeddings, labels_tensor)
            loss.backward()
            optimizer.step()
            loss_arr.append(loss.item())

        elapsed = time.time() - start_time
        epoch_times.append(elapsed)
        scheduler.step()
        print(f"Epoch {epoch + 1} train loss {np.mean(loss_arr):.4f} ({elapsed:.1f}s)")

        # Save checkpoint
        print(f"Saving checkpoint at epoch {epoch} with loss {np.mean(loss_arr):.4f}")
        save_checkpoint(epoch, net, optimizer, loss, args.checkpoint)

        # Validation
        if val_rooms and epoch % args.val_step == args.val_step - 1:
            net.eval()
            val_loss_arr = []

            for i in tqdm(range(len(val_rooms)), desc=f"Epoch {epoch + 1} val"):
                room_feats = val_rooms[i]
                room_labels = val_labels[i]

                if len(room_feats) > args.max_points:
                    subset = np.random.choice(
                        len(room_feats), args.max_points, replace=False
                    )
                    room_feats = room_feats[subset]
                    room_labels = room_labels[subset]

                data = build_ptv3_input(room_feats, args.feature_size, device)
                labels_tensor = torch.LongTensor(room_labels).to(device)

                with torch.no_grad():
                    embeddings = net(data)
                    val_loss = criterion(embeddings, labels_tensor)
                    val_loss_arr.append(val_loss.item())

            print(f"Epoch {epoch + 1} validation loss {np.mean(val_loss_arr):.4f}")

    print(f"\nAvg epoch time: {np.mean(epoch_times):.3f}s")


if __name__ == '__main__':
    main()
