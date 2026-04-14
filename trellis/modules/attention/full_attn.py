from typing import *
import torch
import math
import os
from . import DEBUG, BACKEND

if BACKEND == 'xformers':
    import xformers.ops as xops
elif BACKEND == 'flash_attn':
    import flash_attn
elif BACKEND == 'sdpa':
    from torch.nn.functional import scaled_dot_product_attention as sdpa
elif BACKEND == 'naive':
    pass
else:
    raise ValueError(f"Unknown attention backend: {BACKEND}")


__all__ = [
    'scaled_dot_product_attention',
]

# def modify_attn_score(
#     attn_score: torch.Tensor,
#     reuse_penalty: float = 0.2,
#     max_iter: int | None = 10,
# ) -> torch.Tensor:
#     """
#     Iterative conflict-resolved score modification.

#     Args:
#         attn_score: [B, H, Nq, Nk]
#             Raw attention logits before softmax, i.e. q @ k^T / sqrt(d).
#         reuse_penalty: float
#             Penalty ratio. Loser score on the currently selected key will be reduced by:
#                 reuse_penalty * winner_score_on_that_key
#         max_iter: int or None
#             Max conflict-resolution rounds. If None, defaults to Nk.

#     Returns:
#         modified_score: [B, H, Nq, Nk]
#             Score tensor after iterative conflict resolution.
#     """
#     assert attn_score.dim() == 4, "attn_score must be [B, H, Nq, Nk]"

#     orig_dtype = attn_score.dtype
#     score = attn_score.float().clone()

#     B, H, Nq, Nk = score.shape
#     if max_iter is None:
#         max_iter = Nk

#     # Flatten B and H to simplify vectorized processing
#     M = B * H
#     score = score.reshape(M, Nq, Nk)  # [M, Nq, Nk]

#     device = score.device
#     neg_inf = torch.tensor(float("-inf"), device=device, dtype=score.dtype)

#     for _ in range(max_iter):
#         # Step 1: each query selects its current best key
#         best_val, best_key = score.max(dim=-1)   # [M, Nq], [M, Nq]

#         # Only positive-current-best queries participate
#         active = best_val > 0                    # [M, Nq]
#         if not active.any():
#             break

#         # Step 2: for each key column, find the winner proposal value among current proposers
#         # winner_val_per_key[m, k] = max best_val of queries that currently choose key k
#         winner_val_per_key = torch.full(
#             (M, Nk), neg_inf, device=device, dtype=score.dtype
#         )
#         winner_val_per_key.scatter_reduce_(
#             dim=1,
#             index=best_key,
#             src=torch.where(active, best_val, neg_inf),
#             reduce="amax",
#             include_self=True,
#         )  # [M, Nk]

#         # Gather winner value for each query's currently selected key
#         winner_val_for_query = winner_val_per_key.gather(1, best_key)  # [M, Nq]

#         # Step 3: loser = active and strictly lower than winner on that same selected key
#         # note: ties are kept as co-winners in this version
#         loser = active & (best_val < winner_val_for_query)

#         if not loser.any():
#             break

#         # Step 4: only modify the loser score at its CURRENT selected key
#         loser_idx = loser.nonzero(as_tuple=False)   # [L, 2] => (m, q)
#         m_idx = loser_idx[:, 0]
#         q_idx = loser_idx[:, 1]
#         k_idx = best_key[m_idx, q_idx]

#         cur_val = score[m_idx, q_idx, k_idx]
#         win_val = winner_val_for_query[m_idx, q_idx]

#         # subtract winner-based penalty, but do not cross below 0
#         new_val = torch.clamp(cur_val - reuse_penalty * win_val, min=0.0)

#         score[m_idx, q_idx, k_idx] = new_val

#     score = score.reshape(B, H, Nq, Nk).to(orig_dtype)
#     return score
def modify_attn_score(
    attn_score: torch.Tensor,
    lambda_scale: float = 0.3,
    max_passes: int = 4,
    stop_conflict: float = 0.5,
):
    """
    Fast top-1 conflict reduction on attention logits.

    Args:
        attn_score: [B, H, Lq, Lk]
        lambda_scale: penalty strength
        max_passes: max refinement rounds
        stop_conflict: early stop threshold, e.g. 1.0 or 0.5

    Returns:
        score: [B, H, Lq, Lk]
    """
    orig_dtype = attn_score.dtype
    score = attn_score.float().clone()

    B, H, Lq, Lk = score.shape
    M = B * H
    score = score.reshape(M, Lq, Lk)   # [M, Lq, Lk]

    device = score.device
    neg_inf = torch.tensor(float("-inf"), device=device, dtype=score.dtype)

    for _ in range(max_passes):
        # each query selects its current top-1 key
        best_val, best_key = score.max(dim=-1)   # [M, Lq], [M, Lq]

        # count how many queries choose each key
        counts = torch.zeros(M, Lk, device=device, dtype=score.dtype)
        ones = torch.ones_like(best_val)
        counts.scatter_add_(dim=1, index=best_key, src=ones)   # [M, Lk]

        # hard conflict only
        overload = torch.relu(counts - 1.0)                    # [M, Lk]
        mean_conflict = overload.sum(dim=1).mean()             # scalar

        # early stop
        if mean_conflict.item() <= stop_conflict:
            break

        # winner value on each key column
        winner_val_per_key = torch.full((M, Lk), neg_inf, device=device, dtype=score.dtype)
        winner_val_per_key.scatter_reduce_(
            dim=1,
            index=best_key,
            src=best_val,
            reduce="amax",
            include_self=True,
        )   # [M, Lk]

        winner_val_for_query = winner_val_per_key.gather(1, best_key)   # [M, Lq]
        count_for_query = counts.gather(1, best_key)                    # [M, Lq]
        overload_for_query = overload.gather(1, best_key)               # [M, Lq]

        # loser: selected a crowded key, not the best responder on that key, and current score > 0
        loser = (count_for_query > 1) & (best_val < winner_val_for_query) & (best_val > 0)

        if not loser.any():
            break

        loser_idx = loser.nonzero(as_tuple=False)   # [N, 2]
        m_idx = loser_idx[:, 0]
        q_idx = loser_idx[:, 1]
        k_idx = best_key[m_idx, q_idx]

        cur_val = score[m_idx, q_idx, k_idx]
        win_val = winner_val_for_query[m_idx, q_idx]
        crowd = overload_for_query[m_idx, q_idx]

        # stronger penalty when this key is more crowded
        penalty = lambda_scale * (1.0 + crowd) * win_val

        # positive values after subtraction cannot go below 0
        new_val = torch.clamp(cur_val - penalty, min=0.0)

        changed = new_val < cur_val
        if not changed.any():
            break

        score[m_idx[changed], q_idx[changed], k_idx[changed]] = new_val[changed]

    score = score.reshape(B, H, Lq, Lk).to(orig_dtype)
    return score
# def _naive_sdpa(q, k, v):
#     """
#     Naive implementation of scaled dot product attention.
#     """
#     q = q.permute(0, 2, 1, 3)   # [N, H, L, C]
#     k = k.permute(0, 2, 1, 3)   # [N, H, L, C]
#     v = v.permute(0, 2, 1, 3)   # [N, H, L, C]
#     scale_factor = 1 / math.sqrt(q.size(-1))
#     attn_weight = q @ k.transpose(-2, -1) * scale_factor
#     attn_weight = torch.softmax(attn_weight, dim=-1)
#     out = attn_weight @ v
#     out = out.permute(0, 2, 1, 3)   # [N, L, H, C]
#     return out
def _naive_sdpa(q, k, v, return_score=False, score_reduce="mean", lambda_scale=5.0, enable_truncation=True, modify=True):
    """
    Naive scaled dot product attention.

    Args:
        q: [B, Lq, H, C]
        k: [B, Lk, H, C]
        v: [B, Lk, H, C]
        return_score: whether to return fused token-level score
        score_reduce: how to reduce over key dimension, default "mean"
        lambda_scale: reserved scale factor for later use

    Returns:
        if return_score is False:
            out: [B, Lq, H, C]
        else:
            out: [B, Lq, H, C]
            fused_score: [B, Lq]
            head_weight: [B, Lq, H]
    """
    # -> [B, H, Lq/Lk, C]
    q = q.permute(0, 2, 1, 3)   # [B, H, Lq, C]
    k = k.permute(0, 2, 1, 3)   # [B, H, Lk, C]
    v = v.permute(0, 2, 1, 3)   # [B, H, Lk, C]

    scale_factor = 1.0 / math.sqrt(q.size(-1))

    # [B, H, Lq, Lk]
    attn_logits = torch.matmul(q, k.transpose(-2, -1)) * scale_factor
    if modify:
        attn_logits = modify_attn_score(attn_logits)
    attn_weight = torch.softmax(attn_logits, dim=-1)

    # [B, H, Lq, C]
    out = attn_weight @ v

    # 先保留 per-head 输出，给后面算 head_weight 用
    out_for_weight = out.permute(0, 2, 1, 3)   # [B, Lq, H, C]

    # 最终正常输出
    out = out_for_weight   # [B, Lq, H, C]
    if not return_score:
        return out

    # -------------------------
    # 1) 用每个 head 的输出强度生成 head_weight
    #    [B, Lq, H]
    # -------------------------
    head_strength = out_for_weight.norm(dim=-1)
    head_weight = torch.softmax(head_strength, dim=-1)

    # -------------------------
    # 2) 先把 attn_weight 在 key 维压成每个 query/head 一个分数
    #    [B, H, Lq]
    # -------------------------
    if score_reduce == "mean":
        score_per_head = attn_weight.mean(dim=-1)
    elif score_reduce == "max":
        score_per_head = attn_weight.max(dim=-1).values
    else:
        raise ValueError(f"Unsupported score_reduce: {score_reduce}")

    # -> [B, Lq, H]
    score_per_head = score_per_head.permute(0, 2, 1)
    if enable_truncation:
        # 1. 计算注意力信息熵 H = -sum(p * log(p))
        # 加上 1e-8 防止出现 log(0) 导致的 NaN 崩溃
        # entropy 形状: [B, H, Lq, 1]
        entropy = -(attn_weight * torch.log(attn_weight + 1e-8)).sum(dim=-1, keepdim=True)
        max_logits = attn_logits.max(dim=-1, keepdim=True)[0]
        
        # 2. 设定信息熵阈值
        # 提示: 如果 Lk (序列长度) 是 1024，理论最大熵为 ln(1024) ≈ 6.93
        # 设定一个较高的阈值（如 5.5 或 6.0），只有当分布极其平坦、完全瞎抓时才会拦截
        truncation_threshold = 6.0 
        truncation_max_threshold = 1.0
        # 判定条件：熵大于阈值，说明注意力极度发散，发生了语义错位
        mask_entropy = entropy > truncation_threshold
        mask_max = max_logits < truncation_max_threshold
        mask = mask_entropy & mask_max
        mask = mask.permute(0, 2, 1, 3)
        # 4. 执行暴力置零，依靠残差结构保护当前 3D 拓扑
        if mask is not None:
            out = out.masked_fill(mask, 0.0)
    fused_score = (score_per_head * head_weight).sum(dim=-1)

    return out, fused_score 



@overload
def scaled_dot_product_attention(qkv: torch.Tensor) -> torch.Tensor:
    """
    Apply scaled dot product attention.

    Args:
        qkv (torch.Tensor): A [N, L, 3, H, C] tensor containing Qs, Ks, and Vs.
    """
    ...

@overload
def scaled_dot_product_attention(q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
    """
    Apply scaled dot product attention.

    Args:
        q (torch.Tensor): A [N, L, H, C] tensor containing Qs.
        kv (torch.Tensor): A [N, L, 2, H, C] tensor containing Ks and Vs.
    """
    ...

@overload
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Apply scaled dot product attention.

    Args:
        q (torch.Tensor): A [N, L, H, Ci] tensor containing Qs.
        k (torch.Tensor): A [N, L, H, Ci] tensor containing Ks.
        v (torch.Tensor): A [N, L, H, Co] tensor containing Vs.

    Note:
        k and v are assumed to have the same coordinate map.
    """
    ...

def scaled_dot_product_attention(*args, **kwargs):
    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    return_score = kwargs.get("return_score", False)
    num_all_args = len(args) # + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        assert len(qkv.shape) == 5 and qkv.shape[2] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, L, 3, H, C]"
        device = qkv.device

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
        assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
        device = q.device

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
        assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
        assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
        device = q.device    
    if return_score:
        out, prob_list = _naive_sdpa(q,k,v,return_score=True)
        return out, prob_list

    else:
        if BACKEND == 'xformers':
            if num_all_args == 1:
                q, k, v = qkv.unbind(dim=2)
            elif num_all_args == 2:
                k, v = kv.unbind(dim=2)
            out = xops.memory_efficient_attention(q, k, v)
        elif BACKEND == 'flash_attn':
            if num_all_args == 1:
                out = flash_attn.flash_attn_qkvpacked_func(qkv)
            elif num_all_args == 2:
                out = flash_attn.flash_attn_kvpacked_func(q, kv)
            elif num_all_args == 3:
                out = flash_attn.flash_attn_func(q, k, v)
        elif BACKEND == 'sdpa':
            if num_all_args == 1:
                q, k, v = qkv.unbind(dim=2)
            elif num_all_args == 2:
                k, v = kv.unbind(dim=2)
            q = q.permute(0, 2, 1, 3)   # [N, H, L, C]
            k = k.permute(0, 2, 1, 3)   # [N, H, L, C]
            v = v.permute(0, 2, 1, 3)   # [N, H, L, C]
            out = sdpa(q, k, v)         # [N, H, L, C]
            out = out.permute(0, 2, 1, 3)   # [N, L, H, C]
        elif BACKEND == 'naive':
            if num_all_args == 1:
                q, k, v = qkv.unbind(dim=2)
            elif num_all_args == 2:
                k, v = kv.unbind(dim=2)
            out = _naive_sdpa(q, k, v)
        else:
            raise ValueError(f"Unknown attention module: {BACKEND}")
    
        return out
