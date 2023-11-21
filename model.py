#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import glob
import h5py
import copy
import math
import json
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from util import transform_point_cloud, npmat2euler, quat2mat, visualize_pcs
import open3d as o3d

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k) # q * k^T / sqrt(d_k), sqrt(d_k)用于归一化，防止梯度爆炸
    if mask is not None: # 一直都是None
        scores = scores.masked_fill(mask==0, -1e9)
    p_attn = F.softmax(scores, dim=-1) # 平时attn内部用的是正常softmax
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def pairwise_distance(src, tgt):
    inner = -2 * torch.matmul(src.transpose(2, 1).contiguous(), tgt)
    xx = torch.sum(src**2, dim=1, keepdim=True)
    yy = torch.sum(tgt**2, dim=1, keepdim=True)
    distances = xx.transpose(2, 1).contiguous() + inner + yy
    return torch.sqrt(distances)


def knn(x, k): # 得到每个点的k个最近邻点的index
    inner = -2 * torch.matmul(x.transpose(2, 1).contiguous(), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    distance = -xx - inner - xx.transpose(2, 1).contiguous()

    # idx = distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k) idx = distance.topk(k=k, dim=-1, largest=False)[1] ? 
    values, indices = torch.sort(distance, dim=-1, descending=True) # descending=False? 应该是true, 越远越好？
    idx = indices[..., :k]
    return idx


def get_graph_feature(x, k=20): # 好像是选了20个最近的，然后再从最近的里面找了一个最远的？
    # x = x.squeeze()
    x = x.view(*x.size()[:3]) # [B, 3 (x,y,z), N] 本来就有三个维度，所以没变化
    idx = knn(x, k=k)  # (batch_size, num_points, k)
    batch_size, num_points, _ = idx.size() # [B, N, k=20]
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points # [B, 1, 1], 首先B是针对每个GPU，如果是1无所谓，是2的话那idx_base就是[0, D=768]，因为不同batch knn结果不同，需要区分
    idx = idx + idx_base # 不同B的idx都不一样了

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature, x), dim=3).permute(0, 3, 1, 2) # [B, D', N, k=20]
    return feature


def cycle_consistency(rotation_ab, translation_ab, rotation_ba, translation_ba):
    batch_size = rotation_ab.size(0)
    identity = torch.eye(3, device=rotation_ab.device).unsqueeze(0).repeat(batch_size, 1, 1)
    return F.mse_loss(torch.matmul(rotation_ab, rotation_ba), identity) + F.mse_loss(translation_ab, -translation_ba)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed # 空
        self.tgt_embed = tgt_embed # 空
        self.generator = generator # 空

    def forward(self, src, tgt, src_mask, tgt_mask): 
        # 第一次传入 (src, tgt, None, None)，输出是tgt_embedding
        # 第二次传入 (tgt, src, None, None)，输出是src_embedding
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                           tgt, tgt_mask) # 先encode再decode

    def encode(self, src, src_mask): # mask=None，self.src_embed不就是nn.Sequential里面什么都没写？
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask): # memory=self.encode(src, src_mask)，就是encoder输出作为decoder输入
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))


class Generator(nn.Module): # 这东西都没被调用
    def __init__(self, n_emb_dims):
        super(Generator, self).__init__()
        self.nn = nn.Sequential(nn.Linear(n_emb_dims, n_emb_dims//2),
                                nn.BatchNorm1d(n_emb_dims//2),
                                nn.ReLU(),
                                nn.Linear(n_emb_dims//2, n_emb_dims//4),
                                nn.BatchNorm1d(n_emb_dims//4),
                                nn.ReLU(),
                                nn.Linear(n_emb_dims//4, n_emb_dims//8),
                                nn.BatchNorm1d(n_emb_dims//8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(n_emb_dims//8, 4)
        self.proj_trans = nn.Linear(n_emb_dims//8, 3)

    def forward(self, x):
        x = self.nn(x.max(dim=1)[0])
        rotation = self.proj_rot(x)
        translation = self.proj_trans(x)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        return rotation, translation


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N) # layer就是EncoderLayer, EncoderLayer这个子层又重复了N次
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask): # mask = None
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2



class EncoderLayer(nn.Module): # 用来初始化encoder的各个层
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.norm1 = LayerNorm(size)
        self.dropout1 = nn.Dropout(dropout)
        self.self_attn = self_attn # 指定attention的类型, 是MHA，然后是在Transformer类里面指定了heads和dims
        self.norm2 = LayerNorm(size)
        self.dropout2 = nn.Dropout(dropout)
        self.feed_forward = feed_forward # position-wise feed forward
        self.size = size

    def forward(self, x, mask): # MHA + FFN
        x_norm1 = self.norm1(x)
        x = x + self.self_attn(x_norm1, x_norm1, x_norm1, mask) # attention和feed forward各有一层残差连接
        x_norm2 = self.norm2(x)
        x = x + self.feed_forward(x_norm2)
        return x


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.norm1 = LayerNorm(size)
        self.dropout1 = nn.Dropout(dropout)
        self.self_attn = self_attn
        self.norm2 = LayerNorm(size)
        self.dropout2 = nn.Dropout(dropout)
        self.src_attn = src_attn
        self.norm3 = LayerNorm(size)
        self.dropout3 = nn.Dropout(dropout)
        self.feed_forward = feed_forward

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        x_norm1 = self.norm1(x)
        x = x + self.self_attn(x_norm1, x_norm1, x_norm1, tgt_mask)
        m = memory
        x_norm2 = self.norm2(x)
        x = x + self.src_attn(x_norm2, m, m, src_mask) # tgt做了self attn后作为q, 原始的encoder输出作为k,v
        x_norm3 = self.norm3(x)
        x = x + self.feed_forward(x_norm3)
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.0):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h # dim per head
        self.h = h # h = 4
        self.linears = clones(nn.Linear(d_model, d_model), 4) # 4层linear层，就是矩阵乘法x*w
        self.attn = None # 有什么用吗
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None: # 好像一直都是mask为None
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        result = []
        for linear_layer, input_tensor in zip(self.linears, (query, key, value)): # 就是前三个linear层对q, k, v变换一下
            transformed = linear_layer(input_tensor)
            reshaped_and_transposed = transformed.view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous() # [B, N, D] -> [B, h, N, d], d = D / h
            result.append(reshaped_and_transposed)
        query, key, value = result
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout) # self.attn没什么用

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k) # [B, h, N, d] -> [B, N, h, d] -> [B, N, h*d] = [B, N, D]
        return self.linears[-1](x) # 这里用到了第四层lienar层


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff) # 512 -> 1024
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.leaky_relu(self.w_1(x), negative_slope=0.2)))


class PointNet(nn.Module):
    def __init__(self, n_emb_dims=512):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, n_emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(n_emb_dims)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        return x


class DGCNN(nn.Module):
    def __init__(self, n_emb_dims=512):
        super(DGCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(64*2, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(64*2, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv2d(128*2, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv2d(512, n_emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(n_emb_dims)

    def forward(self, x):
        batch_size, num_dims, num_points = x.size()
        x = get_graph_feature(x)
        x = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.2)
        x1 = x.max(dim=-1, keepdim=True)[0]

        x = get_graph_feature(x1)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x2 = x.max(dim=-1, keepdim=True)[0]

        x = get_graph_feature(x2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x3 = x.max(dim=-1, keepdim=True)[0]

        x = get_graph_feature(x3)
        x = F.leaky_relu(self.bn4(self.conv4(x)), negative_slope=0.2)
        x4 = x.max(dim=-1, keepdim=True)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = F.leaky_relu(self.bn5(self.conv5(x)), negative_slope=0.2).view(batch_size, -1, num_points)
        return x


class MLPHead(nn.Module):
    def __init__(self, args):
        super(MLPHead, self).__init__()
        n_emb_dims = args.n_emb_dims
        self.n_emb_dims = n_emb_dims
        self.nn = nn.Sequential(nn.Linear(n_emb_dims*2, n_emb_dims//2),
                                nn.BatchNorm1d(n_emb_dims//2),
                                nn.ReLU(),
                                nn.Linear(n_emb_dims//2, n_emb_dims//4),
                                nn.BatchNorm1d(n_emb_dims//4),
                                nn.ReLU(),
                                nn.Linear(n_emb_dims//4, n_emb_dims//8),
                                nn.BatchNorm1d(n_emb_dims//8),
                                nn.ReLU())
        self.proj_rot = nn.Linear(n_emb_dims//8, 4)
        self.proj_trans = nn.Linear(n_emb_dims//8, 3)

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        embedding = torch.cat((src_embedding, tgt_embedding), dim=1)
        embedding = self.nn(embedding.max(dim=-1)[0])
        rotation = self.proj_rot(embedding)
        rotation = rotation / torch.norm(rotation, p=2, dim=1, keepdim=True)
        translation = self.proj_trans(embedding)
        return quat2mat(rotation), translation


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, *input):
        return input


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.n_emb_dims = args.n_emb_dims
        self.N = args.n_blocks
        self.dropout = args.dropout
        self.n_ff_dims = args.n_ff_dims
        self.n_heads = args.n_heads
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.n_emb_dims)
        ff = PositionwiseFeedForward(self.n_emb_dims, self.n_ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.n_emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.n_emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous() # tgt_embedding里面有src->tgt的内容
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding


class TemperatureNet(nn.Module):
    def __init__(self, args):
        super(TemperatureNet, self).__init__()
        self.n_emb_dims = args.n_emb_dims
        self.temp_factor = args.temp_factor
        self.nn = nn.Sequential(nn.Linear(self.n_emb_dims, 128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                nn.Linear(128, 128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                nn.Linear(128, 128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                nn.Linear(128, 1),
                                nn.ReLU())
        self.feature_disparity = None

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src_embedding = src_embedding.mean(dim=2) # [B, D, N] -> [B, D]
        tgt_embedding = tgt_embedding.mean(dim=2)
        residual = torch.abs(src_embedding-tgt_embedding) # 如果这时候准确的话，那么他们的差距应该非常小，接近0，如果接近0，就可以给出很高的置信度，让softmax÷λ的λ很小，变成1 0 encoding
        # 这个参数是在训练的时候定的？虽然nn会根据不同的数据有不同结果，但是输出出来基本上一致，那可以假定训练的时候基本上已经fix这个λ (0.8233), 0.8233 < 1说明置信度高于正常
        self.feature_disparity = residual
        return torch.clamp(self.nn(residual), 1.0/self.temp_factor, 1.0*self.temp_factor), residual


class SVDHead(nn.Module):
    def __init__(self, args):
        super(SVDHead, self).__init__()
        self.n_emb_dims = args.n_emb_dims
        self.cat_sampler = args.cat_sampler
        self.reflect = nn.Parameter(torch.eye(3), requires_grad=False)
        self.reflect[2, 2] = -1
        self.temperature = nn.Parameter(torch.ones(1)*0.5, requires_grad=True)
        self.my_iter = torch.ones(1)

    def forward(self, *input):
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        batch_size, num_dims, num_points = src.size() # [B, D, N]
        temperature = input[4].view(batch_size, 1, 1)

        if self.cat_sampler == 'softmax':
            d_k = src_embedding.size(1)
            scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
            scores = torch.softmax(temperature*scores, dim=2)
        elif self.cat_sampler == 'gumbel_softmax':
            d_k = src_embedding.size(1)
            scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k) # 包含相互信息的src_e和tgt_e做了一次attn score计算。相当于q ^ k^T / sqrt(d_k)
            scores = scores.view(batch_size*num_points, num_points)
            temperature = temperature.repeat(1, num_points, 1).view(-1, 1) # [B, num, 1] -> [B*num, 1]
            scores = F.gumbel_softmax(scores, tau=temperature, hard=True) # F.gumbel_softmax输入维度必须是2维
            scores = scores.view(batch_size, num_points, num_points) # 因为softmax对行做，这意味着softmax后，src每个的每个点所代表的行里面，tgt和这个点关系很大的点的概率会很大，其他点的概率会很小
        else:
            raise Exception('not implemented')

        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous()) # 相当于 v * score，其实就得到了src这张图里面特征点在tgt这张图里面的对应点

        src_centered = src - src.mean(dim=2, keepdim=True) # 去中心化

        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)

        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous()).cpu() # src这张图中的特征点x_src， tgt这张图中的x_src' 注意不是tgt，是x_src猜测的点x_src'

        R = []

        # 利用Singular Value Decomposition (SVD)来分解单应矩阵=H=USV^T。
        # 计算旋转矩阵R。首先通过R=VU^T得到一个初始的旋转矩阵。然后通过检查其行列式值来确保它是一个合法的旋转矩阵（行列式值为1）。如果行列式值为-1，那么通过修改一个对角线元素的符号来修正它。
        for i in range(src.size(0)): # B，每个batch（不同的src tgt对）肯定有不同的R T
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0)).contiguous()
            r_det = torch.det(r).item()
            diag = torch.from_numpy(np.array([[1.0, 0, 0],
                                              [0, 1.0, 0],
                                              [0, 0, r_det]]).astype('float32')).to(v.device)
            r = torch.matmul(torch.matmul(v, diag), u.transpose(1, 0)).contiguous() # [3, 3]
            R.append(r)
        
        R = torch.stack(R, dim=0).cuda() # [B, 3, 3]

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True) # R反推T
        if self.training:
            self.my_iter += 1
        return R, t.view(batch_size, 3)


class KeyPointNet(nn.Module):
    def __init__(self, num_keypoints):
        super(KeyPointNet, self).__init__()
        self.num_keypoints = num_keypoints

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src_embedding = input[2]
        tgt_embedding = input[3]
        batch_size, num_dims, num_points = src_embedding.size() # [B, D, N] = [\, 512, 768]
        src_norm = torch.norm(src_embedding, dim=1, keepdim=True) # [\, 1, N]，默认求2范数，比如第一个就是第三维度的第一列的512个数的norm，有点像除去了B这个维度之后对剩下的做batchnorm
        tgt_norm = torch.norm(tgt_embedding, dim=1, keepdim=True)
        src_topk_idx = torch.topk(src_norm, k=self.num_keypoints, dim=2, sorted=False)[1] # num_keypoints=512， 选出最大的512个维度，从768中选512, [512]
        tgt_topk_idx = torch.topk(tgt_norm, k=self.num_keypoints, dim=2, sorted=False)[1]
        src_keypoints_idx = src_topk_idx.repeat(1, 3, 1) # [B, 3, 512]
        tgt_keypoints_idx = tgt_topk_idx.repeat(1, 3, 1)
        src_embedding_idx = src_topk_idx.repeat(1, num_dims, 1) # [B, 512, 512]
        tgt_embedding_idx = tgt_topk_idx.repeat(1, num_dims, 1)

        src_keypoints = torch.gather(src, dim=2, index=src_keypoints_idx)
        tgt_keypoints = torch.gather(tgt, dim=2, index=tgt_keypoints_idx)
        src_embedding = torch.gather(src_embedding, dim=2, index=src_embedding_idx)
        tgt_embedding = torch.gather(tgt_embedding, dim=2, index=tgt_embedding_idx)
        return src_keypoints, tgt_keypoints, src_embedding, tgt_embedding # src本身和src_embedding都被Keypoints筛选好


class ACPNet(nn.Module):
    def __init__(self, args):
        super(ACPNet, self).__init__()
        self.n_emb_dims = args.n_emb_dims
        self.num_keypoints = args.n_keypoints
        self.num_subsampled_points = args.n_subsampled_points
        self.logger = Logger(args)
        if args.emb_nn == 'pointnet':
            self.emb_nn = PointNet(n_emb_dims=self.n_emb_dims)
        elif args.emb_nn == 'dgcnn':
            self.emb_nn = DGCNN(n_emb_dims=self.n_emb_dims)
        else:
            raise Exception('Not implemented')

        if args.attention == 'identity':
            self.attention = Identity()
        elif args.attention == 'transformer':
            self.attention = Transformer(args=args)
        else:
            raise Exception("Not implemented")

        self.temp_net = TemperatureNet(args)

        if args.head == 'mlp':
            self.head = MLPHead(args=args)
        elif args.head == 'svd':
            self.head = SVDHead(args=args)
        else:
            raise Exception('Not implemented')

        if self.num_keypoints != self.num_subsampled_points:
            self.keypointnet = KeyPointNet(num_keypoints=self.num_keypoints)
        else:
            self.keypointnet = Identity()
 
    def forward(self, *input):
        src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity = self.predict_embedding(*input)
        rotation_ab, translation_ab = self.head(src_embedding, tgt_embedding, src, tgt, temperature) # 默认svd head
        rotation_ba, translation_ba = self.head(tgt_embedding, src_embedding, tgt, src, temperature)
        return rotation_ab, translation_ab, rotation_ba, translation_ba, feature_disparity

    def predict_embedding(self, *input): # 传入的input是(src, tgt)
        src = input[0] # 拆分
        tgt = input[1]
        src_embedding = self.emb_nn(src) # 默认使用DGCNN
        tgt_embedding = self.emb_nn(tgt) # 同样参数做src tgt的embedding

        src_embedding_p, tgt_embedding_p = self.attention(src_embedding, tgt_embedding) # 得到了包含相互信息的src_embedding_p和tgt_embedding_p

        src_embedding = src_embedding + src_embedding_p # 残差连接
        tgt_embedding = tgt_embedding + tgt_embedding_p

        src, tgt, src_embedding, tgt_embedding = self.keypointnet(src, tgt, src_embedding, tgt_embedding) # 根据L2范数筛选出512个keypoints

        temperature, feature_disparity = self.temp_net(src_embedding, tgt_embedding)
        return src, tgt, src_embedding, tgt_embedding, temperature, feature_disparity # 返回的Src/tgt及其embedding都是经过筛选的，这里是[1, 3, 768->512]

    def predict_keypoint_correspondence(self, *input):
        src, tgt, src_embedding, tgt_embedding, temperature, _ = self.predict_embedding(*input)
        batch_size, num_dims, num_points = src.size()
        d_k = src_embedding.size(1)
        scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
        scores = scores.view(batch_size*num_points, num_points)
        temperature = temperature.repeat(1, num_points, 1).view(-1, 1)
        scores = F.gumbel_softmax(scores, tau=temperature, hard=True)
        scores = scores.view(batch_size, num_points, num_points)
        return src, tgt, scores


class PRNet(nn.Module):
    def __init__(self, args):
        super(PRNet, self).__init__()
        self.num_iters = args.n_iters
        self.logger = Logger(args)
        self.discount_factor = args.discount_factor
        self.acpnet = ACPNet(args)
        self.model_path = args.model_path
        self.feature_alignment_loss = args.feature_alignment_loss
        self.cycle_consistency_loss = args.cycle_consistency_loss
        self.vis = args.vis 

        if self.model_path is not '':
            print("load pretrain model from:", self.model_path)
            self.load(self.model_path)
        if torch.cuda.device_count() > 1:
            self.acpnet = nn.DataParallel(self.acpnet)

    def forward(self, *input):
        rotation_ab, translation_ab, rotation_ba, translation_ba, feature_disparity = self.acpnet(*input)
        return rotation_ab, translation_ab, rotation_ba, translation_ba, feature_disparity

    def predict(self, src, tgt, n_iters=3):
        batch_size = src.size(0)
        rotation_ab_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)
        for i in range(n_iters):
            rotation_ab_pred_i, translation_ab_pred_i, rotation_ba_pred_i, translation_ba_pred_i, _ \
                = self.forward(src, tgt)
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ab_pred_i
            src = transform_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)

        return rotation_ab_pred, translation_ab_pred

    def _train_one_batch(self, src, tgt, rotation_ab, translation_ab, opt):
        opt.zero_grad()
        batch_size = src.size(0)
        identity = torch.eye(3, device=src.device).unsqueeze(0).repeat(batch_size, 1, 1)

        rotation_ab_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        rotation_ba_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ba_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        total_loss = 0
        total_feature_alignment_loss = 0
        total_cycle_consistency_loss = 0
        total_scale_consensus_loss = 0
        for i in range(self.num_iters):
            rotation_ab_pred_i, translation_ab_pred_i, rotation_ba_pred_i, translation_ba_pred_i, \
            feature_disparity = self.forward(src, tgt)
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ab_pred_i

            rotation_ba_pred = torch.matmul(rotation_ba_pred_i, rotation_ba_pred)
            translation_ba_pred = torch.matmul(rotation_ba_pred_i, translation_ba_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ba_pred_i

            loss = (F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                   + F.mse_loss(translation_ab_pred, translation_ab)) * self.discount_factor**i
            # print(loss)
            feature_alignment_loss = feature_disparity.mean() * self.feature_alignment_loss * self.discount_factor**i
            cycle_consistency_loss = cycle_consistency(rotation_ab_pred_i, translation_ab_pred_i,
                                                       rotation_ba_pred_i, translation_ba_pred_i) \
                                     * self.cycle_consistency_loss * self.discount_factor**i
            scale_consensus_loss = 0
            total_feature_alignment_loss += feature_alignment_loss
            total_cycle_consistency_loss += cycle_consistency_loss
            total_loss = total_loss + loss + feature_alignment_loss + cycle_consistency_loss + scale_consensus_loss
            src = transform_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)
        total_loss.backward()
        opt.step()
        return total_loss.item(), total_feature_alignment_loss.item(), total_cycle_consistency_loss.item(), \
               total_scale_consensus_loss, rotation_ab_pred, translation_ab_pred

    def _test_one_batch(self, src, tgt, rotation_ab, translation_ab):
        batch_size = src.size(0)
        identity = torch.eye(3, device=src.device).unsqueeze(0).repeat(batch_size, 1, 1)

        rotation_ab_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ab_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        rotation_ba_pred = torch.eye(3, device=src.device, dtype=torch.float32).view(1, 3, 3).repeat(batch_size, 1, 1)
        translation_ba_pred = torch.zeros(3, device=src.device, dtype=torch.float32).view(1, 3).repeat(batch_size, 1)

        total_loss = 0
        total_feature_alignment_loss = 0
        total_cycle_consistency_loss = 0
        total_scale_consensus_loss = 0
        for i in range(self.num_iters):
            rotation_ab_pred_i, translation_ab_pred_i, rotation_ba_pred_i, translation_ba_pred_i, \
            feature_disparity = self.forward(src, tgt)
            rotation_ab_pred = torch.matmul(rotation_ab_pred_i, rotation_ab_pred)
            translation_ab_pred = torch.matmul(rotation_ab_pred_i, translation_ab_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ab_pred_i

            rotation_ba_pred = torch.matmul(rotation_ba_pred_i, rotation_ba_pred)
            translation_ba_pred = torch.matmul(rotation_ba_pred_i, translation_ba_pred.unsqueeze(2)).squeeze(2) \
                                  + translation_ba_pred_i

            loss = (F.mse_loss(torch.matmul(rotation_ab_pred.transpose(2, 1), rotation_ab), identity) \
                    + F.mse_loss(translation_ab_pred, translation_ab)) * self.discount_factor ** i
            feature_alignment_loss = feature_disparity.mean() * self.feature_alignment_loss * self.discount_factor ** i
            cycle_consistency_loss = cycle_consistency(rotation_ab_pred_i, translation_ab_pred_i,
                                                       rotation_ba_pred_i, translation_ba_pred_i) \
                                     * self.cycle_consistency_loss * self.discount_factor ** i
            scale_consensus_loss = 0
            total_feature_alignment_loss += feature_alignment_loss
            total_cycle_consistency_loss += cycle_consistency_loss
            total_loss = total_loss + loss + feature_alignment_loss + cycle_consistency_loss + scale_consensus_loss
            src = transform_point_cloud(src, rotation_ab_pred_i, translation_ab_pred_i)
        return total_loss.item(), total_feature_alignment_loss.item(), total_cycle_consistency_loss.item(), \
               total_scale_consensus_loss, rotation_ab_pred, translation_ab_pred

    def _train_one_epoch(self, epoch, train_loader, opt): # 先调用这个train_one_epoch
        self.train()
        total_loss = 0
        rotations_ab = []
        translations_ab = []
        rotations_ab_pred = []
        translations_ab_pred = []
        eulers_ab = []
        num_examples = 0
        total_feature_alignment_loss = 0.0
        total_cycle_consistency_loss = 0.0
        total_scale_consensus_loss = 0.0
        for data in tqdm(train_loader): # data_loader中本来就包含了这类数据
            src, tgt, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba = [d.cuda()
                                                                                                      for d in data]
            loss, feature_alignment_loss, cycle_consistency_loss, scale_consensus_loss,\
            rotation_ab_pred, translation_ab_pred = self._train_one_batch(src, tgt, rotation_ab, translation_ab,
                                                                                opt)
            batch_size = src.size(0)
            num_examples += batch_size
            total_loss = total_loss + loss * batch_size
            total_feature_alignment_loss = total_feature_alignment_loss + feature_alignment_loss * batch_size
            total_cycle_consistency_loss = total_cycle_consistency_loss + cycle_consistency_loss * batch_size
            total_scale_consensus_loss = total_scale_consensus_loss + scale_consensus_loss * batch_size

            rotations_ab.append(rotation_ab.detach().cpu().numpy())
            translations_ab.append(translation_ab.detach().cpu().numpy())
            rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
            translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
            eulers_ab.append(euler_ab.cpu().numpy())
        avg_loss = total_loss / num_examples
        avg_feature_alignment_loss = total_feature_alignment_loss / num_examples
        avg_cycle_consistency_loss = total_cycle_consistency_loss / num_examples
        avg_scale_consensus_loss = total_scale_consensus_loss / num_examples

        rotations_ab = np.concatenate(rotations_ab, axis=0)
        translations_ab = np.concatenate(translations_ab, axis=0)
        rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
        translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
        eulers_ab = np.degrees(np.concatenate(eulers_ab, axis=0))
        eulers_ab_pred = npmat2euler(rotations_ab_pred)
        r_ab_mse = np.mean((eulers_ab-eulers_ab_pred)**2)
        r_ab_rmse = np.sqrt(r_ab_mse)
        r_ab_mae = np.mean(np.abs(eulers_ab-eulers_ab_pred))
        t_ab_mse = np.mean((translations_ab-translations_ab_pred)**2)
        t_ab_rmse = np.sqrt(t_ab_mse)
        t_ab_mae = np.mean(np.abs(translations_ab-translations_ab_pred))
        r_ab_r2_score = r2_score(eulers_ab, eulers_ab_pred)
        t_ab_r2_score = r2_score(translations_ab, translations_ab_pred)
        info = {'arrow': 'A->B',
                'epoch': epoch,
                'stage': 'train',
                'loss': avg_loss,
                'feature_alignment_loss': avg_feature_alignment_loss,
                'cycle_consistency_loss': avg_cycle_consistency_loss,
                'scale_consensus_loss': avg_scale_consensus_loss,
                'r_ab_mse': r_ab_mse,
                'r_ab_rmse': r_ab_rmse,
                'r_ab_mae': r_ab_mae,
                't_ab_mse': t_ab_mse,
                't_ab_rmse': t_ab_rmse,
                't_ab_mae': t_ab_mae,
                'r_ab_r2_score': r_ab_r2_score,
                't_ab_r2_score': t_ab_r2_score}
        self.logger.write(info)
        return info

    


    def _test_one_epoch(self, epoch, test_loader):
        self.eval()
        total_loss = 0
        rotations_ab = []
        translations_ab = []
        rotations_ab_pred = []
        translations_ab_pred = []
        eulers_ab = []

        original_pcs = []
        transformed_pcs = []

        num_examples = 0
        total_feature_alignment_loss = 0.0
        total_cycle_consistency_loss = 0.0
        total_scale_consensus_loss = 0.0

        for data in tqdm(test_loader):
            src, tgt, rotation_ab, translation_ab, rotation_ba, translation_ba, euler_ab, euler_ba = [d.cuda()
                                                                                                      for d in data]
            original_pcs.append(src.detach().cpu().numpy())
            loss, feature_alignment_loss, cycle_consistency_loss, scale_consensus_loss, \
            rotation_ab_pred, translation_ab_pred = self._test_one_batch(src, tgt, rotation_ab, translation_ab)
            batch_size = src.size(0)
            num_examples += batch_size
            total_loss = total_loss + loss * batch_size
            total_feature_alignment_loss = total_feature_alignment_loss + feature_alignment_loss * batch_size
            total_cycle_consistency_loss = total_cycle_consistency_loss + cycle_consistency_loss * batch_size
            total_scale_consensus_loss = total_scale_consensus_loss + scale_consensus_loss * batch_size

            rotations_ab.append(rotation_ab.detach().cpu().numpy())
            translations_ab.append(translation_ab.detach().cpu().numpy())
            rotations_ab_pred.append(rotation_ab_pred.detach().cpu().numpy())
            translations_ab_pred.append(translation_ab_pred.detach().cpu().numpy())
            eulers_ab.append(euler_ab.cpu().numpy())

            transformed_pc = transform_point_cloud(src.detach(), rotation_ab_pred, translation_ab_pred)
            transformed_pcs.append(transformed_pc.detach().cpu().numpy())
        
        if self.vis: 
            for idx, (original_pc, transformed_pc) in enumerate(zip(original_pcs, transformed_pcs)):
                original_o3d = o3d.geometry.PointCloud()
                original_o3d.points = o3d.utility.Vector3dVector(original_pc[0])  

                transformed_o3d = o3d.geometry.PointCloud()
                transformed_o3d.points = o3d.utility.Vector3dVector(transformed_pc[0])  

                filename = str(idx) + 'screencapture'
                visualize_pcs(original_o3d, transformed_o3d, filename)
                print(f'Epoch {idx} is visualized')

        avg_loss = total_loss / num_examples
        avg_feature_alignment_loss = total_feature_alignment_loss / num_examples
        avg_cycle_consistency_loss = total_cycle_consistency_loss / num_examples
        avg_scale_consensus_loss = total_scale_consensus_loss / num_examples

        rotations_ab = np.concatenate(rotations_ab, axis=0)
        translations_ab = np.concatenate(translations_ab, axis=0)
        rotations_ab_pred = np.concatenate(rotations_ab_pred, axis=0)
        translations_ab_pred = np.concatenate(translations_ab_pred, axis=0)
        eulers_ab = np.degrees(np.concatenate(eulers_ab, axis=0))
        eulers_ab_pred = npmat2euler(rotations_ab_pred)
        r_ab_mse = np.mean((eulers_ab - eulers_ab_pred) ** 2)
        r_ab_rmse = np.sqrt(r_ab_mse)
        r_ab_mae = np.mean(np.abs(eulers_ab - eulers_ab_pred))
        t_ab_mse = np.mean((translations_ab - translations_ab_pred) ** 2)
        t_ab_rmse = np.sqrt(t_ab_mse)
        t_ab_mae = np.mean(np.abs(translations_ab - translations_ab_pred))
        r_ab_r2_score = r2_score(eulers_ab, eulers_ab_pred)
        t_ab_r2_score = r2_score(translations_ab, translations_ab_pred)

        info = {'arrow': 'A->B',
                'epoch': epoch,
                'stage': 'test',
                'loss': avg_loss,
                'feature_alignment_loss': avg_feature_alignment_loss,
                'cycle_consistency_loss': avg_cycle_consistency_loss,
                'scale_consensus_loss': avg_scale_consensus_loss,
                'r_ab_mse': r_ab_mse,
                'r_ab_rmse': r_ab_rmse,
                'r_ab_mae': r_ab_mae,
                't_ab_mse': t_ab_mse,
                't_ab_rmse': t_ab_rmse,
                't_ab_mae': t_ab_mae,
                'r_ab_r2_score': r_ab_r2_score,
                't_ab_r2_score': t_ab_r2_score}
        self.logger.write(info)
        return info

    def save(self, path):
        if torch.cuda.device_count() > 1:
            torch.save(self.acpnet.module.state_dict(), path)
        else:
            torch.save(self.acpnet.state_dict(), path)

    def load(self, path):
        self.acpnet.load_state_dict(torch.load(path))


class Logger:
    def __init__(self, args):
        self.path = 'checkpoints/' + args.exp_name
        self.fw = open(self.path+'/log', 'a')
        self.fw.write(str(args))
        self.fw.write('\n')
        self.fw.flush()
        print(str(args))
        with open(os.path.join(self.path, 'args.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

    def write(self, info):
        arrow = info['arrow']
        epoch = info['epoch']
        stage = info['stage']
        loss = info['loss']
        feature_alignment_loss = info['feature_alignment_loss']
        cycle_consistency_loss = info['cycle_consistency_loss']
        scale_consensus_loss = info['scale_consensus_loss']
        r_ab_mse = info['r_ab_mse']
        r_ab_rmse = info['r_ab_rmse']
        r_ab_mae = info['r_ab_mae']
        t_ab_mse = info['t_ab_mse']
        t_ab_rmse = info['t_ab_rmse']
        t_ab_mae = info['t_ab_mae']
        r_ab_r2_score = info['r_ab_r2_score']
        t_ab_r2_score = info['t_ab_r2_score']
        text = '%s:: Stage: %s, Epoch: %d, Loss: %f, Feature_alignment_loss: %f, Cycle_consistency_loss: %f, ' \
               'Scale_consensus_loss: %f, Rot_MSE: %f, Rot_RMSE: %f, ' \
               'Rot_MAE: %f, Rot_R2: %f, Trans_MSE: %f, ' \
               'Trans_RMSE: %f, Trans_MAE: %f, Trans_R2: %f\n' % \
               (arrow, stage, epoch, loss, feature_alignment_loss, cycle_consistency_loss, scale_consensus_loss,
                r_ab_mse, r_ab_rmse, r_ab_mae,
                r_ab_r2_score, t_ab_mse, t_ab_rmse, t_ab_mae, t_ab_r2_score)
        self.fw.write(text)
        self.fw.flush()
        print(text)

    def close(self):
        self.fw.close()


if __name__ == '__main__':
    print('hello world')
