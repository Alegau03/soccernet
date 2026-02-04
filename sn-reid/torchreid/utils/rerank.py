#!/usr/bin/env python2/python3
# -*- coding: utf-8 -*-
"""
Source: https://github.com/zhunzhong07/person-re-ranking

Created on Mon Jun 26 14:46:56 2017
@author: luohao
Modified by Houjing Huang, 2017-12-22.
- This version accepts distance matrix instead of raw features.
- The difference of `/` division between python 2 and 3 is handled.
- numpy.float16 is replaced by numpy.float32 for numerical precision.

CVPR2017 paper:Zhong Z, Zheng L, Cao D, et al. Re-ranking Person Re-identification with k-reciprocal Encoding[J]. 2017.
url:http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhong_Re-Ranking_Person_Re-Identification_CVPR_2017_paper.pdf
Matlab version: https://github.com/zhunzhong07/person-re-ranking
"""
from __future__ import division, print_function, absolute_import
import numpy as np

__all__ = ['re_ranking']


def re_ranking(q_g_dist, q_q_dist, g_g_dist, k1=20, k2=6, lambda_value=0.3):

    # Ensure float16 for inputs to save memory
    q_g_dist = q_g_dist.astype(np.float16)
    q_q_dist = q_q_dist.astype(np.float16)
    g_g_dist = g_g_dist.astype(np.float16)

    query_num = q_g_dist.shape[0]
    gallery_num = q_g_dist.shape[1]
    all_num = query_num + gallery_num
    
    search_k = k1 + 1
    
    # Pass 1: Global Scaling Factors (Robust Max per Row)
    # We need max(row i) for all i to perform the normalization d(i,j)/max(j).
    # Since matrix is symmetric, max(row j) == max(col j).
    print("Pass 1: Computing global scaling factors...")
    all_max_vals = np.zeros(all_num, dtype=np.float32)
    
    # Process queries
    for i in range(query_num):
        if np.mod(i, 100) == 0:
            print(f"Pass 1: {i}/{query_num}")
            
        row = np.concatenate([q_q_dist[i], q_g_dist[i]]).astype(np.float32)
        row = np.power(row, 2)
        
        # Reverting to simple Global Max.
        # If we have 5000^2 (25e6) in the row, this becomes the max.
        # Valid distances (e.g. 100) became 100/25e6 ~ 0.
        # exp(-0) = 1.
        # This makes the weights of all valid neighbors ~1 (Uniform).
        # This acts as a robust "Voting" mechanism (counting shared neighbors)
        # instead of weighing them by noisy distance.
        all_max_vals[i] = np.max(row)


    # Process gallery
    q_g_dist_T = q_g_dist.T
    for i in range(gallery_num):
        row = np.concatenate([q_g_dist_T[i], g_g_dist[i]]).astype(np.float32)
        row = np.power(row, 2)
        all_max_vals[query_num + i] = np.max(row)
            
    print("Pass 2: Computing initial ranks with local scaling...")
    
    initial_rank = np.zeros((all_num, search_k), dtype=np.int32)
    
    # Process queries
    for i in range(query_num):
        row = np.concatenate([q_q_dist[i], q_g_dist[i]]).astype(np.float32)
        row = np.power(row, 2)
        
        # Scaling: d(i, j) / all_max_vals[j]
        # This changes the sort order compared to raw distances!
        row = row / all_max_vals
        
        # Argpartition is robust to NaNs but here we have cleaned data
        idx = np.argpartition(row, search_k)[:search_k]
        idx = idx[np.argsort(row[idx])]
        initial_rank[i] = idx

    # Process gallery
    for i in range(gallery_num):
        row = np.concatenate([q_g_dist_T[i], g_g_dist[i]]).astype(np.float32)
        row = np.power(row, 2)
        
        row = row / all_max_vals
        
        idx = np.argpartition(row, search_k)[:search_k]
        idx = idx[np.argsort(row[idx])]
        initial_rank[query_num + i] = idx
        
    print("Initial ranking computed. Starting k-reciprocal expansion...")
    
    # Expansion weights are exp(-original_dist/max_vals).
    # Since we already computed scaled row in pass 2, we can reuse logic or recompute locally.
    # Recomputing locally is cheap for sparse set.
    
    def get_scaled_sq_dist(i, j):
        val = 0.0
        if i < query_num:
            if j < query_num:
                val = q_q_dist[i, j]
            else:
                val = q_g_dist[i, j - query_num]
        else:
            i_g = i - query_num
            if j < query_num:
                val = q_g_dist_T[i_g, j]
            else:
                val = g_g_dist[i_g, j - query_num]
        
        val = val ** 2
        # Normalize by max_vals[j]? No, paper says: 
        # weight = exp(-d(q,g)^2) if normalized globally.
        # But wait, logic for V: weight = exp(-original_dist[i, idx]).
        # original_dist was the normalized matrix.
        # So yes, we need scaled distance.
        if all_max_vals[j] > 0:
            val /= all_max_vals[j]
        return val

    V_sparse = [{} for _ in range(all_num)]

    for i in range(all_num):
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[
                candidate, :int(np.around(k1 / 2.)) + 1]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index, :int(np.around(k1 / 2.)) + 1]
            fi_candidate = np.where(
                candidate_backward_k_neigh_index == candidate
            )[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[
                fi_candidate]
            if len(
                np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)
            ) > 2. / 3 * len(candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index
                )

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        
        vals = []
        for idx in k_reciprocal_expansion_index:
             vals.append(get_scaled_sq_dist(i, idx))
        
        vals = np.array(vals, dtype=np.float32)
        weight = np.exp(-vals)
        norm_weight = weight / np.sum(weight)
        
        for idx, w in zip(k_reciprocal_expansion_index, norm_weight):
            V_sparse[i][idx] = w

    if k2 != 1:
        V_qe_sparse = [{} for _ in range(all_num)]
        for i in range(all_num):
            neighbors = initial_rank[i, :k2]
            accum = {}
            for n_idx in neighbors:
                 for k, v in V_sparse[n_idx].items():
                     accum[k] = accum.get(k, 0.0) + v
            
            for k, sum_v in accum.items():
                V_qe_sparse[i][k] = sum_v / k2
        
        V_sparse = V_qe_sparse
        del V_qe_sparse

    del initial_rank

    print("Building inverted index for Jaccard...")
    invIndex = {} 
    for i in range(query_num, all_num):
        for n_idx in V_sparse[i].keys():
            if n_idx not in invIndex:
                invIndex[n_idx] = []
            invIndex[n_idx].append(i)
            
    print("Computing Jaccard distances...")
    final_dist = np.zeros((query_num, gallery_num), dtype=np.float32)
    
    for i in range(query_num):
        temp_min = {} 
        for n_idx, val_i in V_sparse[i].items():
            if n_idx in invIndex:
                for g_idx in invIndex[n_idx]:
                    val_j = V_sparse[g_idx][n_idx]
                    term = min(val_i, val_j)
                    temp_min[g_idx] = temp_min.get(g_idx, 0.0) + term
        
        # Original Dist term
        row_orig = q_g_dist[i].astype(np.float32)
        row_orig = np.power(row_orig, 2)
        
        # We need to normalize this row by the GLOBAL SCALING logic: d(i,j)/max(j)
        # We have all_max_vals[query_num : query_num+gallery_num] responsible for the gallery columns
        gallery_max_vals = all_max_vals[query_num:]
        
        # Safe divide
        # Replace 0s with 1s to avoid nan
        safe_div = np.where(gallery_max_vals > 1e-6, gallery_max_vals, 1.0)
        row_orig = row_orig / safe_div
            
        final_dist[i, :] = row_orig * lambda_value + (1 - lambda_value)
        
        for g_idx, t_min in temp_min.items():
            g_loc = g_idx - query_num
            jaccard_val = 1 - t_min / (2 - t_min)
            
            final_dist[i, g_loc] = jaccard_val * (1 - lambda_value) + row_orig[g_loc] * lambda_value

    return final_dist
