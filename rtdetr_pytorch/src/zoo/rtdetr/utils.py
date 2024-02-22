"""by lyuwenyu
"""

import math
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.onnx.symbolic_helper import parse_args, _get_tensor_dim_size, _get_tensor_sizes


def inverse_sigmoid(x: torch.Tensor, eps: float=1e-5) -> torch.Tensor:
    x = x.clip(min=0., max=1.)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def deformable_attention_core_func(value, value_spatial_shapes, sampling_locations, attention_weights):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[level].flatten(2).permute(
            0, 2, 1).reshape(bs * n_head, c, h, w)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(
            0, 2, 1, 3, 4).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points)
    output = (torch.stack(
        sampling_value_list, dim=-2).flatten(-2) *
              attention_weights).sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)

class MultiScaleDeformableAttnFunction_pytorch(torch.nn.Module):
    @staticmethod
    def symbolic(g, value, value_spatial_shapes, sampling_locations, attention_weights):
        output = g.op('com.microsoft::MultiScaleDeformableAttn',value, value_spatial_shapes, sampling_locations, attention_weights)
        
        bs, _, mum_heads, embed_dims_num_heads = _get_tensor_sizes(value)
        bs, num_queries, _, _, _, _ = _get_tensor_sizes(sampling_locations)
        output_shape = [bs, num_queries, mum_heads * embed_dims_num_heads]
        output.setType(value.type().with_sizes(output_shape))

        return output

    @staticmethod
    def forward(self, value, value_spatial_shapes, sampling_locations, attention_weights):
        bs, _, n_head, c = value.shape
        _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

        split_shape = [h * w for h, w in value_spatial_shapes]
        value_list = value.split(split_shape, dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for level, (h, w) in enumerate(value_spatial_shapes):
            # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
            value_l_ = value_list[level].flatten(2).permute(
                0, 2, 1).reshape(bs * n_head, c, h, w)
            # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
            sampling_grid_l_ = sampling_grids[:, :, :, level].permute(
                0, 2, 1, 3, 4).flatten(0, 1)
            # N_*M_, D_, Lq_, P_
            sampling_value_l_ = F.grid_sample(
                value_l_,
                sampling_grid_l_,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False)
            sampling_value_list.append(sampling_value_l_)
        # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
        attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
            bs * n_head, 1, Len_q, n_levels * n_points)
        output = (torch.stack(
            sampling_value_list, dim=-2).flatten(-2) *
                attention_weights).sum(-1).reshape(bs, n_head * c, Len_q)

        return output.permute(0, 2, 1)


class MultiScaleDeformableAttnFunction_pytorch_op(torch.autograd.Function):
    @staticmethod
    def symbolic(g, value, value_spatial_shapes, sampling_locations, attention_weights):
        output = g.op('com.microsoft::MultiScaleDeformableAttn',value, value_spatial_shapes, sampling_locations, attention_weights)
        
        bs, _, mum_heads, embed_dims_num_heads = _get_tensor_sizes(value)
        bs, num_queries, _, _, _, _ = _get_tensor_sizes(sampling_locations)
        output_shape = [bs, num_queries, mum_heads * embed_dims_num_heads]
        output.setType(value.type().with_sizes(output_shape))

        return output

    @staticmethod
    def forward(self, value, value_spatial_shapes, sampling_locations, attention_weights):
        bs, _, n_head, c = value.shape
        _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

        split_shape = [h * w for h, w in value_spatial_shapes]
        value_list = value.split(split_shape, dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampling_value_list = []
        for level, (h, w) in enumerate(value_spatial_shapes):
            # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
            value_l_ = value_list[level].flatten(2).permute(
                0, 2, 1).reshape(bs * n_head, c, h, w)
            # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
            sampling_grid_l_ = sampling_grids[:, :, :, level].permute(
                0, 2, 1, 3, 4).flatten(0, 1)
            # N_*M_, D_, Lq_, P_
            sampling_value_l_ = F.grid_sample(
                value_l_,
                sampling_grid_l_,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=False)
            sampling_value_list.append(sampling_value_l_)
        # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
        attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
            bs * n_head, 1, Len_q, n_levels * n_points)
        output = (torch.stack(
            sampling_value_list, dim=-2).flatten(-2) *
                attention_weights).sum(-1).reshape(bs, n_head * c, Len_q)

        return output.permute(0, 2, 1)


# class MultiScaleDeformableAttnFunction_trt(torch.autograd.Function):
#     @staticmethod
#     def symbolic(g, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights):
#         output = g.op('com.microsoft::MultiscaleDeformableAttnPlugin_TRT',value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        
#         bs, _, mum_heads, embed_dims_num_heads = _get_tensor_sizes(value)
#         bs, num_queries, _, _, _, _ = _get_tensor_sizes(sampling_locations)
#         output_shape = [bs, num_queries, mum_heads * embed_dims_num_heads]
#         output.setType(value.type().with_sizes(output_shape))

#         return output

#     @staticmethod
#     def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights):
#         bs, _, n_head, c = value.shape
#         _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

#         split_shape = [h * w for h, w in value_spatial_shapes]
#         value_list = value.split(split_shape, dim=1)
#         sampling_grids = 2 * sampling_locations - 1
#         sampling_value_list = []
#         for level, (h, w) in enumerate(value_spatial_shapes):
#             # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
#             value_l_ = value_list[level].flatten(2).permute(
#                 0, 2, 1).reshape(bs * n_head, c, h, w)
#             # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
#             sampling_grid_l_ = sampling_grids[:, :, :, level].permute(
#                 0, 2, 1, 3, 4).flatten(0, 1)
#             # N_*M_, D_, Lq_, P_
#             sampling_value_l_ = F.grid_sample(
#                 value_l_,
#                 sampling_grid_l_,
#                 mode='bilinear',
#                 padding_mode='zeros',
#                 align_corners=False)
#             sampling_value_list.append(sampling_value_l_)
#         # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
#         attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
#             bs * n_head, 1, Len_q, n_levels * n_points)
#         output = (torch.stack(
#             sampling_value_list, dim=-2).flatten(-2) *
#                 attention_weights).sum(-1).reshape(bs, n_head * c, Len_q)

#         return output.permute(0, 2, 1)


import math 
def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init



def get_activation(act: str, inpace: bool=True):
    '''get activation
    '''
    act = act.lower()
    
    if act == 'silu':
        m = nn.SiLU()

    elif act == 'relu':
        m = nn.ReLU()

    elif act == 'leaky_relu':
        m = nn.LeakyReLU()

    elif act == 'silu':
        m = nn.SiLU()
    
    elif act == 'gelu':
        m = nn.GELU()
        
    elif act is None:
        m = nn.Identity()
    
    elif isinstance(act, nn.Module):
        m = act

    else:
        raise RuntimeError('')  

    if hasattr(m, 'inplace'):
        m.inplace = inpace
    
    return m 


