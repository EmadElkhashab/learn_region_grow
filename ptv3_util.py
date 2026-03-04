import numpy
from model import *
import h5py
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from point_transformer_pytorch import PointTransformerLayer

def save_checkpoint(epoch, model, optimizer, loss, filepath):
	checkpoint = {
		'epoch': epoch,
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'loss': loss,
	}
	torch.save(checkpoint, filepath)

class RegionTransformerPTv3(nn.Module):
	# Updated to use PointTransformerV3
	def __init__(self, batch_size=32,
				 seq_len=1,
				 num_inlier_points=512,
				 num_neighbour_points=512,
				 feature_dim=12):
		super(RegionTransformerPTv3, self).__init__()
		self.num_points = num_inlier_points
		self.batch_size = batch_size
		
		self.ptv3 = PointTransformerV3(
			in_channels=feature_dim-6,
			order=("z", "z-trans", "hilbert", "hilbert-trans"),
			#enable_rpe=True,
			#enable_flash=False,
			stride=(2, 2, 2, 2),
			enc_depths=(2, 2, 2, 6, 2),
			enc_channels=(32, 64, 128, 256, 512),
			enc_num_head=(2, 4, 8, 16, 32),
			enc_patch_size=(128, 128, 128, 128, 128),
			dec_depths=(2, 2, 2, 2),
			dec_channels=(128, 128, 256, 256),
			dec_num_head=(4, 4, 8, 16),
			dec_patch_size=(128, 128, 128, 128),
		)
		
		self.remove_mask = nn.Linear(128, 1)  
		self.add_mask = nn.Linear(128, 1)  
		
	def forward(self, inlier_points, neighbour_points):

		inlier_feats = inlier_points['feat'].unsqueeze(0).reshape(self.batch_size, self.num_points, -1)
		inlier_coords = inlier_points['coord'].unsqueeze(0).reshape(self.batch_size, self.num_points, -1)
		neighbour_feats = neighbour_points['feat'].unsqueeze(0).reshape(self.batch_size, self.num_points, -1)
		neighbour_coords = neighbour_points['coord'].unsqueeze(0).reshape(self.batch_size, self.num_points, -1)
		
		all_points_feat = torch.cat([inlier_feats, neighbour_feats], dim=1).reshape(self.batch_size*self.num_points*2, -1)
		all_points_coords = torch.cat([inlier_coords, neighbour_coords], dim=1).reshape(self.batch_size*self.num_points*2, -1)
		all_points_offsets = torch.tensor([(i+1)*self.num_points*2 for i in range(self.batch_size)]).to("cuda")
		all_points_grid_size = 0.01 # Customizable
		all_points = {
			'feat': all_points_feat,
			'coord': all_points_coords,
			'offset': all_points_offsets,
			'grid_size': all_points_grid_size
		}

		x = self.ptv3(all_points)
		features = x['feat'].reshape(self.batch_size, self.num_points*2, -1)
		remove_feats = features[:, :self.num_points, :]
		add_feats = features[:, self.num_points:, :]

		remove_mask_logits = self.remove_mask(remove_feats).squeeze()
		add_mask_logits = self.add_mask(add_feats).squeeze()

		return remove_mask_logits, add_mask_logits