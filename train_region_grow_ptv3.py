from ptv3_util import *
import os
import sys
import time
import numpy
import h5py
import torch
import torch.nn as nn
from model import *
from tqdm import tqdm

MODEL_PATH = None
BATCH_SIZE = 32
NUM_INLIER_POINT = 512
NUM_NEIGHBOR_POINT = 512
CURRENT_EPOCH = 3
MAX_EPOCH = 64
TRANSFORMER = False
VAL_STEP = 7
TRAIN_AREA = ['scannet']#,'2','3','scannet','4','5','6']
VAL_AREA = None
#	Set to 12 as curvatures may be nan in scan data.
FEATURE_SIZE = 12
MULTISEED = 0

initialized = False
cross_domain = False
numpy.random.seed(0)
numpy.set_printoptions(2,linewidth=100,suppress=True,sign=' ')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

net = RegionTransformerPTv3(BATCH_SIZE, 1, NUM_INLIER_POINT, NUM_NEIGHBOR_POINT, FEATURE_SIZE)
net = net.to(device)
torch.compile(net)

optimizer = torch.optim.AdamW(net.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=64, eta_min=0)

if CURRENT_EPOCH != 0:
	checkpoint = torch.load("model_checkpoint.pth")
	net.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	print(f"Loaded checkpoint from epoch {CURRENT_EPOCH}")

epoch_time = []

for epoch in range(CURRENT_EPOCH, MAX_EPOCH):

	if not initialized or MULTISEED > 1:
		initialized = True
		train_inlier_points, train_inlier_count, train_neighbor_points, train_neighbor_count, train_add, train_remove = [], [], [], [], [], []
		val_inlier_points, val_inlier_count, val_neighbor_points, val_neighbor_count, val_add, val_remove = [], [], [], [], [], []

		if VAL_AREA is not None and epoch % VAL_STEP == VAL_STEP - 1:
			AREA_LIST = TRAIN_AREA + VAL_AREA
		else:
			AREA_LIST = TRAIN_AREA
		for AREA in AREA_LIST:
			if isinstance(AREA, str) and AREA.startswith('synthetic'):
				f = h5py.File('data/staged_%s.h5' % AREA, 'r')
			elif MULTISEED > 0 and AREA in TRAIN_AREA:
				SEED = epoch % MULTISEED
				try:
					f = h5py.File('data/multiseed/seed%d_area%s.h5'%(SEED,AREA),'r')
				except OSError:
					print("OSError")
					continue
			else:
				f = h5py.File('data/staged_area%s.h5'%(AREA),'r')
			print('Loading %s ...'%f.filename)
			if VAL_AREA is not None and AREA in VAL_AREA:
				count = f['count'][:]
				val_inlier_count.extend(count)
				points = f['points'][:]
				remove = f['remove'][:]
				idp = 0
				for i in range(len(count)):
					val_inlier_points.append(points[idp:idp+count[i], :FEATURE_SIZE])
					val_remove.append(remove[idp:idp+count[i]])
					idp += count[i]
				neighbor_count = f['neighbor_count'][:]
				val_neighbor_count.extend(neighbor_count)
				neighbor_points = f['neighbor_points'][:]
				add = f['add'][:]
				idp = 0
				for i in range(len(neighbor_count)):
					val_neighbor_points.append(neighbor_points[idp:idp+neighbor_count[i], :FEATURE_SIZE])
					val_add.append(add[idp:idp+neighbor_count[i]])
					idp += neighbor_count[i]
			if AREA in TRAIN_AREA:
				count = f['count'][:]
				train_inlier_count.extend(count)
				points = f['points'][:]
				remove = f['remove'][:]
				idp = 0
				for i in range(len(count)):
					train_inlier_points.append(points[idp:idp+count[i], :FEATURE_SIZE])
					train_remove.append(remove[idp:idp+count[i]])
					idp += count[i]
				neighbor_count = f['neighbor_count'][:]
				train_neighbor_count.extend(neighbor_count)
				neighbor_points = f['neighbor_points'][:]
				add = f['add'][:]
				idp = 0
				for i in range(len(neighbor_count)):
					train_neighbor_points.append(neighbor_points[idp:idp+neighbor_count[i], :FEATURE_SIZE])
					train_add.append(add[idp:idp+neighbor_count[i]])
					idp += neighbor_count[i]
			if FEATURE_SIZE is None:
				FEATURE_SIZE = points.shape[1]
			f.close()

		#filter out instances where the neighbor array is empty
		train_inlier_points = [train_inlier_points[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_inlier_count = [train_inlier_count[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_neighbor_points = [train_neighbor_points[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_add = [train_add[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_remove = [train_remove[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		train_neighbor_count = [train_neighbor_count[i] for i in range(len(train_neighbor_count)) if train_neighbor_count[i]>0]
		val_inlier_points = [val_inlier_points[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_inlier_count = [val_inlier_count[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_neighbor_points = [val_neighbor_points[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_add = [val_add[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_remove = [val_remove[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		val_neighbor_count = [val_neighbor_count[i] for i in range(len(val_neighbor_count)) if val_neighbor_count[i]>0]
		if len(train_inlier_points)==0:
			continue
		print('train',len(train_inlier_points),train_inlier_points[0].shape, len(train_neighbor_points))
		print('val',len(val_inlier_points), len(val_neighbor_points))

	idx = numpy.arange(len(train_inlier_points))
	numpy.random.shuffle(idx)
	inlier_points = numpy.zeros((BATCH_SIZE, NUM_INLIER_POINT, FEATURE_SIZE))
	neighbor_points = numpy.zeros((BATCH_SIZE, NUM_NEIGHBOR_POINT, FEATURE_SIZE))
	input_add = numpy.zeros((BATCH_SIZE, NUM_NEIGHBOR_POINT), dtype=numpy.int32)
	input_remove = numpy.zeros((BATCH_SIZE, NUM_INLIER_POINT), dtype=numpy.int32)

	loss_arr = []
	num_batches = int(len(train_inlier_points) / BATCH_SIZE)
	start_time = time.time()
	net.train()
	for batch_id in tqdm(range(num_batches)):
		start_idx = batch_id * BATCH_SIZE
		end_idx = (batch_id + 1) * BATCH_SIZE
		for i in range(BATCH_SIZE):
			points_idx = idx[start_idx+i]
			N = train_inlier_count[points_idx]
			if N >= NUM_INLIER_POINT:
				subset = numpy.random.choice(N, NUM_INLIER_POINT, replace=False)
			else:
				subset = list(range(N)) + list(numpy.random.choice(N, NUM_INLIER_POINT-N, replace=True))
			inlier_points[i,:,:] = train_inlier_points[points_idx][subset, :]
			input_remove[i,:] = train_remove[points_idx][subset]
			N = train_neighbor_count[points_idx]
			if N >= NUM_NEIGHBOR_POINT:
				subset = numpy.random.choice(N, NUM_NEIGHBOR_POINT, replace=False)
			else:
				subset = list(range(N)) + list(numpy.random.choice(N, NUM_NEIGHBOR_POINT-N, replace=True))
			neighbor_points[i,:,:] = train_neighbor_points[points_idx][subset, :]
			input_add[i,:] = train_add[points_idx][subset]

		inlier_tensor = torch.FloatTensor(inlier_points).to(device)
		inlier_normalized = inlier_tensor[:,:,3:6].reshape(-1, 3)
		inlier_feats = inlier_tensor[:,:,6:FEATURE_SIZE].reshape(-1, FEATURE_SIZE-6)
		inlier_offsets = torch.tensor([(i+1)*NUM_INLIER_POINT for i in range(BATCH_SIZE)]).to(device)
		inlier_data = {
				'coord': inlier_normalized,
				'feat': inlier_feats,
				'offset': inlier_offsets,
				'grid_size': 0.01
			}

		neighbor_tensor = torch.FloatTensor(neighbor_points).to(device)
		neighbor_coords = neighbor_tensor[:,:,:3].reshape(-1, 3)
		neighbor_normalized = neighbor_tensor[:,:,3:6].reshape(-1, 3)
		neighbor_feats = neighbor_tensor[:,:,6:FEATURE_SIZE].reshape(-1, FEATURE_SIZE-6)
		neighbor_offsets = torch.tensor([(i+1)*NUM_NEIGHBOR_POINT for i in range(BATCH_SIZE)]).to(device)
		neighbor_data = {
			'coord': neighbor_normalized,
			'feat': neighbor_feats,
			'offset': neighbor_offsets,
			'grid_size': 0.01
		}			
		add_mask_tensor = torch.FloatTensor(input_add).to(device)
		remove_mask_tensor = torch.FloatTensor(input_remove).to(device)

		optimizer.zero_grad()
		remove_mask_logits, add_mask_logits = net(inlier_data, neighbor_data)

		add_loss = criterion(add_mask_logits, add_mask_tensor)
		remove_loss = criterion(remove_mask_logits, remove_mask_tensor)
		loss = add_loss + remove_loss
		loss.backward()
		optimizer.step()

		loss_arr.append(loss.item())
		#print("Batch %d Total_loss %.2f Loss %.2f"%(batch_id, numpy.mean(loss_arr), loss.item()))


	epoch_time.append(time.time() - start_time)
	scheduler.step()

	print("Epoch %d train loss %.2f"%(epoch+1,numpy.mean(loss_arr)))

	if epoch % 1 == 0: # Save every 5 epochs
		print(f"Saving checkpoint at epoch {epoch} with loss {numpy.mean(loss_arr):.4f}")
		save_checkpoint(epoch, net, optimizer, loss, 'model_checkpoint.pth')

	if VAL_AREA is not None and epoch % VAL_STEP == VAL_STEP - 1:
		net.eval()
		loss_arr = []
		num_batches = int(len(val_inlier_points) / BATCH_SIZE)
		for batch_id in tqdm(range(num_batches)):
			start_idx = batch_id * BATCH_SIZE
			end_idx = (batch_id + 1) * BATCH_SIZE
			for i in range(BATCH_SIZE):
				points_idx = start_idx+i
				N = val_inlier_count[points_idx]
				if N >= NUM_INLIER_POINT:
					subset = numpy.random.choice(N, NUM_INLIER_POINT, replace=False)
				else:
					subset = list(range(N)) + list(numpy.random.choice(N, NUM_INLIER_POINT-N, replace=True))
				inlier_points[i,:,:] = val_inlier_points[points_idx][subset, :]
				input_remove[i,:] = val_remove[points_idx][subset]
				N = val_neighbor_count[points_idx]
				if N >= NUM_INLIER_POINT:
					subset = numpy.random.choice(N, NUM_NEIGHBOR_POINT, replace=False)
				else:
					subset = list(range(N)) + list(numpy.random.choice(N, NUM_NEIGHBOR_POINT-N, replace=True))
				neighbor_points[i,:,:] = val_neighbor_points[points_idx][subset, :]
				input_add[i,:] = val_add[points_idx][subset]

			inlier_tensor = torch.FloatTensor(inlier_points).to(device)
			inlier_coords = inlier_tensor[:,:,:3].reshape(-1, 3)
			inlier_normalized = inlier_tensor[:,:,3:6].reshape(-1, 3)
			inlier_feats = inlier_tensor[:,:,6:FEATURE_SIZE].reshape(-1, FEATURE_SIZE-6)
			inlier_offsets = torch.tensor([(i+1)*NUM_INLIER_POINT for i in range(BATCH_SIZE)]).to(device)
			inlier_data = {
						'coord': inlier_normalized,
						'feat': inlier_feats,
						'offset': inlier_offsets,
						'grid_size': 0.01
					}
			
			neighbor_tensor = torch.FloatTensor(neighbor_points).to(device)
			neighbor_coords = neighbor_tensor[:,:,:3].reshape(-1, 3)
			neighbor_normalized = neighbor_tensor[:,:,3:6].reshape(-1, 3)
			neighbor_feats = neighbor_tensor[:,:,6:FEATURE_SIZE].reshape(-1, FEATURE_SIZE-6)
			neighbor_offsets = torch.tensor([(i+1)*NUM_NEIGHBOR_POINT for i in range(BATCH_SIZE)]).to(device)
			neighbor_data = {
				'coord': neighbor_normalized,
				'feat': neighbor_feats,
				'offset': neighbor_offsets,
				'grid_size': 0.01
			}	
			add_mask_tensor = torch.FloatTensor(input_add).to(device)
			remove_mask_tensor = torch.FloatTensor(input_remove).to(device)
			
			with torch.no_grad():
				remove_mask_logits, add_mask_logits = net(inlier_tensor, neighbor_tensor)
				add_loss = criterion(add_mask_logits, add_mask_tensor)
				remove_loss = criterion(remove_mask_logits, remove_mask_tensor)
				ls = add_loss.item() + remove_loss.item()

			loss_arr.append(ls)

		print("Epoch %d validation loss %.2f"%(epoch+1,numpy.mean(loss_arr)))


print("Avg Epoch Time: %.3f" % numpy.mean(epoch_time))