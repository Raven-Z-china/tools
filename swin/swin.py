import math
import numpy as np

import torch
from mmcv.runner import auto_fp16
from torch import nn

from sst_ops import flat2window_v2, window2flat_v2, get_flat2win_inds_v2, get_window_coors, get_inner_win_inds
import random
import pickle as pkl
import os

from mmdet.models.builder import BACKBONES
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry
MODELS = Registry('models', parent=MMCV_MODELS)
MIDDLE_ENCODERS = MODELS

from mmdet.models import BACKBONES
import copy
from mmcv.cnn import build_conv_layer, build_norm_layer, ConvModule
from sst_basic_block_v2 import BasicShiftBlockV2

from mmcv.runner import BaseModule



class Swin():
    def __init__(self, region_shape, grid_size,region_drop_info):
        self.embed_dims=256
        self.get_regions = nn.ModuleList()
        self.grid2region_att = nn.ModuleList()
        for l in range(len(region_shape)):
            this_embed_dim = self.embed_dims//2*(l+1)
            region_metas = dict(
                type='SSTInputLayerV2',
                window_shape=region_shape[l],
                sparse_shape=grid_size[l],
                shuffle_voxels=True,
                drop_info=region_drop_info[l],
                pos_temperature=1000,
                normalize_pos=False,
                pos_embed=this_embed_dim,
            )
            grid2region_att = dict(
                type='SSTv2',
                d_model=[this_embed_dim,] * 4,
                nhead=[8, ] * 4,
                num_blocks=1,
                dim_feedforward=[this_embed_dim, ] * 4,
                output_shape=grid_size[l][:2],
                in_channel=self.embed_dims//2 if l==0 else None,
            )
            self.get_regions.append(MIDDLE_ENCODERS.build(region_metas))
            self.grid2region_att.append(BACKBONES.build(grid2region_att))

        pts_backbone = dict(
            type='SECONDV2',
            in_channels=128,
            out_channels=[128, 256],
            layer_nums=[5, 5],
            layer_strides=[1, 2],
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
            conv_cfg=dict(type='Conv2d', bias=False))
        self.pts_backbone = BACKBONES.build(pts_backbone)

    def create_dense_coord(self, x_size, y_size, batch_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        batch_z = torch.zeros_like(batch_x)
        coord_base = torch.cat([batch_z[None], batch_x[None], batch_y[None]], dim=0)
        batch_coord = []

        for i in range(batch_size):
            batch_idx = torch.ones_like(batch_x)[None] * i
            this_coord_base = torch.cat([batch_idx, coord_base], dim=0)
            batch_coord.append(this_coord_base)

        batch_coord = torch.stack(batch_coord)
        return batch_coord

    def swin_att(self,bev_size,bev_feats,bs):
        grid_features = bev_feats.flatten(2, 3).permute(0, 2, 1).reshape(-1, bev_feats.shape[1])
        bev_coords = self.create_dense_coord(bev_size, bev_size, bs).type_as(grid_features).int()
        this_coords = []
        for k in range(bs):
            this_coord = bev_coords[k].reshape(4, -1).transpose(1, 0)
            this_coords.append(this_coord)
        grid_coords = torch.cat(this_coords, dim=0)
        return_feats = []
        for i in range(len(self.get_regions)):
            x = self.get_regions[i](grid_features, grid_coords, bs)
            x = self.grid2region_att[i](x)

            grid_features, grid_coords, this_feat = self.pts_backbone(x, 'stage{}'.format(i+1))
            return_feats.append(this_feat)

        return return_feats


@MIDDLE_ENCODERS.register_module()
class SSTInputLayerV2(nn.Module):
    """
    This is one of the core class of SST, converting the output of voxel_encoder to sst input.
    There are 3 things to be done in this class:
    1. Reginal Grouping : assign window indices to each voxel.
    2. Voxel drop and region batching: see our paper for detail
    3. Pre-computing the transfomation information for converting flat features ([N x C]) to region features ([R, T, C]).
        R is the number of regions containing at most T tokens (voxels). See function flat2window and window2flat for details.

    Main args:
        drop_info (dict): drop configuration for region batching. 
        window_shape (tuple[int]): (num_x, num_y). Each window is divided to num_x * num_y pillars (including empty pillars).
        shift_list (list[tuple]): [(shift_x, shift_y), ]. shift_x = 5 means all windonws will be shifted for 5 voxels along positive direction of x-aixs.
        debug: apply strong assertion for developing. 
    """

    def __init__(self,
        drop_info,
        window_shape,
        sparse_shape,
        shuffle_voxels=True,
        debug=False,
        normalize_pos=False,
        pos_temperature=10000,
        mute=False,
        pos_embed=256,
        **kwargs,
        ):
        super().__init__()
        self.fp16_enabled = False
        self.meta_drop_info = drop_info
        self.sparse_shape = sparse_shape
        self.shuffle_voxels = shuffle_voxels
        self.debug = debug
        self.window_shape = window_shape
        self.normalize_pos = normalize_pos
        self.pos_temperature = pos_temperature
        self.mute = mute
        self.pos_embed_channels = pos_embed

        self.bev_embedding = None


    @auto_fp16(apply_to=('voxel_feat', ))
    def forward(self, voxel_feats, voxel_coors, batch_size=None):
        '''
        Args:
            voxel_feats: shape=[N, C], N is the voxel num in the batch.
            coors: shape=[N, 4], [b, z, y, x]
        Returns:
            feat_3d_dict: contains region features (feat_3d) of each region batching level. Shape of feat_3d is [num_windows, num_max_tokens, C].
            flat2win_inds_list: two dict containing transformation information for non-shifted grouping and shifted grouping, respectively. The two dicts are used in function flat2window and window2flat.
            voxel_info: dict containing extra information of each voxel for usage in the backbone.
        '''

        self.set_drop_info()
        voxel_coors = voxel_coors.long()

        if self.shuffle_voxels:
            # shuffle the voxels to make the drop process uniform.
            shuffle_inds = torch.randperm(len(voxel_feats))
            voxel_feats = voxel_feats[shuffle_inds]
            voxel_coors = voxel_coors[shuffle_inds]

        voxel_info = self.window_partition(voxel_coors)
        voxel_info['voxel_feats'] = voxel_feats
        voxel_info['voxel_coors'] = voxel_coors
        voxel_info = self.drop_voxel(voxel_info, 2) # voxel_info is updated in this function

        voxel_feats = voxel_info['voxel_feats'] # after dropping
        voxel_coors = voxel_info['voxel_coors']


        for i in range(2):
            voxel_info[f'flat2win_inds_shift{i}'] = \
                get_flat2win_inds_v2(voxel_info[f'batch_win_inds_shift{i}'], voxel_info[f'voxel_drop_level_shift{i}'], self.drop_info, debug=True) 

            voxel_info[f'pos_dict_shift{i}'] = \
                self.get_pos_embed(voxel_info[f'flat2win_inds_shift{i}'], voxel_info[f'coors_in_win_shift{i}'], self.pos_embed_channels, voxel_feats.dtype)

            voxel_info[f'key_mask_shift{i}'] = \
                self.get_key_padding_mask(voxel_info[f'flat2win_inds_shift{i}'])

        if self.debug:
            coors_3d_dict_shift0 = flat2window_v2(voxel_coors, voxel_info['flat2win_inds_shift0'])
            coors_2d = window2flat_v2(coors_3d_dict_shift0, voxel_info['flat2win_inds_shift0'])
            assert (coors_2d == voxel_coors).all()

        if self.shuffle_voxels:
            voxel_info['shuffle_inds'] = shuffle_inds

        return voxel_info
    
    def drop_single_shift(self, batch_win_inds):
        drop_info = self.drop_info
        drop_lvl_per_voxel = -torch.ones_like(batch_win_inds)
        inner_win_inds = get_inner_win_inds(batch_win_inds)   # inner_win_inds:非空voxel在每个window内的位置索引
        # inner_win_inds = self.get_inner_win_inds(batch_win_inds)
        bincount = torch.bincount(batch_win_inds)
        num_per_voxel_before_drop = bincount[batch_win_inds] #
        target_num_per_voxel = torch.zeros_like(batch_win_inds)

        for dl in drop_info:
            max_tokens = drop_info[dl]['max_tokens']
            lower, upper = drop_info[dl]['drop_range']
            range_mask = (num_per_voxel_before_drop >= lower) & (num_per_voxel_before_drop < upper)
            target_num_per_voxel[range_mask] = max_tokens
            drop_lvl_per_voxel[range_mask] = dl
        
        if self.debug:
            assert (target_num_per_voxel > 0).all()
            assert (drop_lvl_per_voxel >= 0).all()

        keep_mask = inner_win_inds < target_num_per_voxel
        return keep_mask, drop_lvl_per_voxel

    def drop_voxel(self, voxel_info, num_shifts):
        '''
        To make it clear and easy to follow, we do not use loop to process two shifts.
        '''

        batch_win_inds_s0 = voxel_info['batch_win_inds_shift0']
        num_all_voxel = batch_win_inds_s0.shape[0]

        voxel_keep_inds = torch.arange(num_all_voxel, device=batch_win_inds_s0.device, dtype=torch.long)

        keep_mask_s0, drop_lvl_s0 = self.drop_single_shift(batch_win_inds_s0)
        if self.debug:
            assert (drop_lvl_s0 >= 0).all()

        drop_lvl_s0 = drop_lvl_s0[keep_mask_s0]
        voxel_keep_inds = voxel_keep_inds[keep_mask_s0]
        batch_win_inds_s0 = batch_win_inds_s0[keep_mask_s0]

        if num_shifts == 1:
            voxel_info['voxel_keep_inds'] = voxel_keep_inds
            voxel_info['voxel_drop_level_shift0'] = drop_lvl_s0
            voxel_info['batch_win_inds_shift0'] = batch_win_inds_s0
            return voxel_info

        batch_win_inds_s1 = voxel_info['batch_win_inds_shift1']
        batch_win_inds_s1 = batch_win_inds_s1[keep_mask_s0]

        keep_mask_s1, drop_lvl_s1 = self.drop_single_shift(batch_win_inds_s1)
        if self.debug:
            assert (drop_lvl_s1 >= 0).all()

        # drop data in first shift again
        drop_lvl_s0 = drop_lvl_s0[keep_mask_s1]
        voxel_keep_inds = voxel_keep_inds[keep_mask_s1]
        batch_win_inds_s0 = batch_win_inds_s0[keep_mask_s1]

        drop_lvl_s1 = drop_lvl_s1[keep_mask_s1]
        batch_win_inds_s1 = batch_win_inds_s1[keep_mask_s1]

        voxel_info['voxel_keep_inds'] = voxel_keep_inds
        voxel_info['voxel_drop_level_shift0'] = drop_lvl_s0
        voxel_info['batch_win_inds_shift0'] = batch_win_inds_s0
        voxel_info['voxel_drop_level_shift1'] = drop_lvl_s1
        voxel_info['batch_win_inds_shift1'] = batch_win_inds_s1
        voxel_keep_inds = voxel_info['voxel_keep_inds']

        voxel_num_before_drop = len(voxel_info['voxel_coors'])
        voxel_info['voxel_feats'] = voxel_info['voxel_feats'][voxel_keep_inds]
        voxel_info['voxel_coors'] = voxel_info['voxel_coors'][voxel_keep_inds]

        # Some other variables need to be dropped.
        for k, v in voxel_info.items():
            if isinstance(v, torch.Tensor) and len(v) == voxel_num_before_drop:
                voxel_info[k] = v[voxel_keep_inds]

        ### sanity check
        if self.debug and self.training:
            for dl in self.drop_info:
                max_tokens = self.drop_info[dl]['max_tokens']

                mask_s0 = drop_lvl_s0 == dl
                if not mask_s0.any():
                    if not self.mute:
                        print(f'No voxel belongs to drop_level:{dl} in shift 0')
                    continue
                real_max = torch.bincount(batch_win_inds_s0[mask_s0]).max()
                assert real_max <= max_tokens, f'real_max({real_max}) > {max_tokens} in shift0'

                mask_s1 = drop_lvl_s1 == dl
                if not mask_s1.any():
                    if not self.mute:
                        print(f'No voxel belongs to drop_level:{dl} in shift 1')
                    continue
                real_max = torch.bincount(batch_win_inds_s1[mask_s1]).max()
                assert real_max <= max_tokens, f'real_max({real_max}) > {max_tokens} in shift1'
        ###
        return voxel_info

    @torch.no_grad()
    def window_partition(self, coors):
        voxel_info = {}
        for i in range(2):
            batch_win_inds, coors_in_win = get_window_coors(coors, self.sparse_shape, self.window_shape, i == 1)
            voxel_info[f'batch_win_inds_shift{i}'] = batch_win_inds
            voxel_info[f'coors_in_win_shift{i}'] = coors_in_win
        
        return voxel_info

    @torch.no_grad()
    def get_pos_embed(self, inds_dict, coors_in_win, feat_dim, dtype):
        '''
        Args:
        coors_in_win: shape=[N, 3], order: z, y, x
        '''

        # [N,]
        window_shape = self.window_shape
        if len(window_shape) == 2:
            ndim = 2
            win_x, win_y = window_shape
            win_z = 0
        elif  window_shape[-1] == 1:
            ndim = 2
            win_x, win_y = window_shape[:2]
            win_z = 0
        else:
            win_x, win_y, win_z = window_shape
            ndim = 3

        assert coors_in_win.size(1) == 3
        z, y, x = coors_in_win[:, 0] - win_z/2, coors_in_win[:, 1] - win_y/2, coors_in_win[:, 2] - win_x/2
        assert (x >= -win_x/2 - 1e-4).all()
        assert (x <= win_x/2-1 + 1e-4).all()

        if self.normalize_pos:
            x = x / win_x * 2 * 3.1415 #[-pi, pi]
            y = y / win_y * 2 * 3.1415 #[-pi, pi]
            z = z / win_z * 2 * 3.1415 #[-pi, pi]
        
        pos_length = feat_dim // ndim
        # [pos_length]
        inv_freq = torch.arange(
            pos_length, dtype=torch.float32, device=coors_in_win.device)
        inv_freq = self.pos_temperature ** (2 * (inv_freq // 2) / pos_length)

        # [num_tokens, pos_length]
        embed_x = x[:, None] / inv_freq[None, :]
        embed_y = y[:, None] / inv_freq[None, :]
        if ndim == 3:
            embed_z = z[:, None] / inv_freq[None, :]

        # [num_tokens, pos_length]
        embed_x = torch.stack([embed_x[:, ::2].sin(), embed_x[:, 1::2].cos()], dim=-1).flatten(1)
        embed_y = torch.stack([embed_y[:, ::2].sin(), embed_y[:, 1::2].cos()], dim=-1).flatten(1)
        if ndim == 3:
            embed_z = torch.stack([embed_z[:, ::2].sin(), embed_z[:, 1::2].cos()], dim=-1).flatten(1)

        # [num_tokens, c]
        if ndim == 3:
            pos_embed_2d = torch.cat([embed_x, embed_y, embed_z], dim=-1).to(dtype)
        else:
            pos_embed_2d = torch.cat([embed_x, embed_y], dim=-1).to(dtype)
        
        gap = feat_dim - pos_embed_2d.size(1)
        assert gap >= 0
        if gap > 0:
            assert ndim == 3
            padding = torch.zeros((pos_embed_2d.size(0), gap), dtype=dtype, device=coors_in_win.device)
            pos_embed_2d = torch.cat([pos_embed_2d, padding], dim=1)
        else:
            assert ndim == 2

        pos_embed_dict = flat2window_v2(
            pos_embed_2d, inds_dict)

        return pos_embed_dict

    @torch.no_grad()
    def get_key_padding_mask(self, ind_dict):
        num_all_voxel = len(ind_dict['voxel_drop_level'])
        key_padding = torch.ones((num_all_voxel, 1)).to(ind_dict['voxel_drop_level'].device).bool()

        window_key_padding_dict = flat2window_v2(key_padding, ind_dict)

        # logical not. True means masked
        for key, value in window_key_padding_dict.items():
            window_key_padding_dict[key] = value.logical_not().squeeze(2)
        
        return window_key_padding_dict

    def set_drop_info(self):
        if hasattr(self, 'drop_info'):
            return
        meta = self.meta_drop_info
        if isinstance(meta, tuple):
            self.drop_info = meta
        else:
            self.drop_info = meta
        print(f'drop_info is set to {self.drop_info}, in input_layer')


@BACKBONES.register_module()
class SSTv2(nn.Module):
    '''
    Single-stride Sparse Transformer. 
    Main args:
        d_model (list[int]): the number of filters in first linear layer of each transformer encoder
        dim_feedforward list([int]): the number of filters in first linear layer of each transformer encoder
        output_shape (tuple[int, int]): shape of output bev feature.
        num_attached_conv: the number of convolutions in the end of SST for filling the "empty hold" in BEV feature map.
        conv_kwargs: key arguments of each attached convolution.
        checckpoint_blocks: block IDs (0 to num_blocks - 1) to use checkpoint.
        Note: In PyTorch 1.8, checkpoint function seems not able to receive dict as parameters. Better to use PyTorch >= 1.9.
    '''

    def __init__(
        self,
        d_model=[],
        nhead=[],
        num_blocks=6,
        dim_feedforward=[],
        dropout=0.0,
        activation="gelu",
        output_shape=None,
        debug=True,
        in_channel=None,
        checkpoint_blocks=[],
        layer_cfg=dict(),
        ):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        self.checkpoint_blocks = checkpoint_blocks

        if in_channel is not None:
            self.linear0 = nn.Linear(in_channel, d_model[0])

        # Sparse Regional Attention Blocks
        block_list=[]
        for i in range(num_blocks):
            block_list.append(
                BasicShiftBlockV2(d_model[i], nhead[i], dim_feedforward[i],
                    dropout, activation, batch_first=False, block_id=i, layer_cfg=layer_cfg)
            )

        self.block_list = nn.ModuleList(block_list)
            
        self._reset_parameters()

        self.output_shape = output_shape

        self.debug = debug


    def forward(self, voxel_info, **kwargs):
        '''
        '''
        num_shifts = 2 
        assert voxel_info['voxel_coors'].dtype == torch.int64, 'data type of coors should be torch.int64!'

        batch_size = voxel_info['voxel_coors'][:, 0].max().item() + 1
        voxel_feat = voxel_info['voxel_feats']
        ind_dict_list = [voxel_info[f'flat2win_inds_shift{i}'] for i in range(num_shifts)]
        padding_mask_list = [voxel_info[f'key_mask_shift{i}'] for i in range(num_shifts)]
        pos_embed_list = [voxel_info[f'pos_dict_shift{i}'] for i in range(num_shifts)]

        output = voxel_feat
        if hasattr(self, 'linear0'):
            output = self.linear0(output)

        for i, block in enumerate(self.block_list):   # sst block
            output = block(output, pos_embed_list, ind_dict_list,
                           padding_mask_list, using_checkpoint = i in self.checkpoint_blocks)

        output = self.recover_bev(output, voxel_info['voxel_coors'], batch_size)

        output_list = []
        output_list.append(output)

        return output_list

    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'scaler' not in name and 'tau' not in name:
                nn.init.xavier_uniform_(p)

    def recover_bev(self, voxel_feat, coors, batch_size):
        '''
        Args:
            voxel_feat: shape=[N, C]
            coors: [N, 4]
        Return:
            batch_canvas:, shape=[B, C, ny, nx]
        '''
        ny, nx = self.output_shape
        feat_dim = voxel_feat.shape[-1]

        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                feat_dim,
                nx * ny,
                dtype=voxel_feat.dtype,
                device=voxel_feat.device)

            # Only include non-empty pillars
            batch_mask = coors[:, 0] == batch_itt
            this_coors = coors[batch_mask, :]
            indices = this_coors[:, 2] * nx + this_coors[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_feat[batch_mask, :] #[n, c]
            voxels = voxels.t() #[c, n]

            canvas[:, indices] = voxels

            batch_canvas.append(canvas)

        batch_canvas = torch.stack(batch_canvas, 0)

        batch_canvas = batch_canvas.view(batch_size, feat_dim, ny, nx)

        return batch_canvas


@BACKBONES.register_module()  # sst mode
class SECONDV2(BaseModule):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(self,
                 in_channels=128,
                 out_channels=[128, 128, 256],
                 layer_nums=[3, 5, 5],
                 layer_strides=[2, 2, 2],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 init_cfg=None,
                 pretrained=None):
        super(SECONDV2, self).__init__(init_cfg=init_cfg)
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            if layer_strides[i] == 2:
                self.ds_layer = nn.Sequential(build_conv_layer(
                    conv_cfg,
                    in_filters[i],
                    out_channels[i],
                    3,
                    stride=layer_strides[i],
                    padding=1),
                    build_norm_layer(norm_cfg, out_channels[i])[1],
                    nn.ReLU(inplace=True),)
                block = []
            else:
                block = [
                    build_conv_layer(
                        conv_cfg,
                        in_filters[i],
                        out_channels[i],
                        3,
                        stride=layer_strides[i],
                        padding=1),
                    build_norm_layer(norm_cfg, out_channels[i])[1],
                    nn.ReLU(inplace=True),
                ]

            for j in range(layer_num):
                block.append(
                    build_conv_layer(
                        conv_cfg,
                        out_channels[i],
                        out_channels[i],
                        3,
                        padding=1))
                block.append(build_norm_layer(norm_cfg, out_channels[i])[1])
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)


        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        else:
            self.init_cfg = dict(type='Kaiming', layer='Conv2d')


    def create_dense_coord(self, x_size, y_size, batch_size):
        meshgrid = [[0, x_size - 1, x_size], [0, y_size - 1, y_size]]
        # NOTE: modified
        batch_x, batch_y = torch.meshgrid(
            *[torch.linspace(it[0], it[1], it[2]) for it in meshgrid]
        )
        # batch_idx =  torch.zeros_like(batch_x)
        batch_z = torch.zeros_like(batch_x)
        coord_base = torch.cat([batch_z[None], batch_x[None], batch_y[None]], dim=0)
        batch_coord = []

        for i in range(batch_size):
            batch_idx = torch.ones_like(batch_x)[None] * i
            this_coord_base = torch.cat([batch_idx, coord_base], dim=0)
            batch_coord.append(this_coord_base)

        batch_coord = torch.stack(batch_coord)
        return batch_coord


    def forward(self, x, stage=None):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        if stage=="stage1":
            x = self.blocks[0](x[0])

            return_features = self.ds_layer(x)
            N, C, H, W = return_features.shape
            return_features = return_features.reshape(N, C, -1).permute(0, 2, 1).reshape(-1, C)
            bev_coords = self.create_dense_coord(H, W, N).type_as(return_features).int()
            this_coords = []
            for k in range(N):
                this_coord = bev_coords[k]
                this_coord = this_coord.reshape(4, -1).transpose(1, 0)
                this_coords.append(this_coord)

            return_coords = torch.cat(this_coords, dim=0)

            return return_features, return_coords, x


        elif stage=="stage2":
            x = self.blocks[1](x[0])
            return None, None, x

        else:
            x_1 = self.blocks[0](x)
            outs.append(x_1)
            x_2 = self.ds_layer(x_1)
            x_2 = self.blocks[1](x_2)
            outs.append(x_2)
            return tuple(outs)

            
if __name__ == '__main__':
    # 定义坐标范围
    x = torch.arange(0, 10)  # x 方向 (0 到 9)
    y = torch.arange(0, 10)  # y 方向 (0 到 9)

    # 使用 meshgrid 生成网格
    xx, yy = torch.meshgrid(x, y)  # 矩阵索引方式

    # 将坐标合并为 (x, y) 对
    coords = torch.stack([xx, yy], dim=-1)  # 形状为 [10, 10, 2]

    # 展平并合并到一行
    print(coords)
    mat=coords[None].permute(0,3,1,2)
    bev_size=180
    grid_size = [[bev_size, bev_size, 1], [bev_size//2, bev_size//2, 1]]
    region_shape = [(6, 6, 1), (6, 6, 1)]
    region_drop_info = [
    {0:{'max_tokens':36, 'drop_range':(0, 100000)},},
    {0:{'max_tokens':36, 'drop_range':(0, 100000)},},
    ]
    swin=Swin(region_shape,grid_size,region_drop_info)
    x=swin.swin_att(bev_size,mat,1)
    print(mat.permute(0,2,3,1))

    