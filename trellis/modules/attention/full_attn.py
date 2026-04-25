from typing import *
import math

import torch

from . import BACKEND

if BACKEND == "xformers":
    import xformers.ops as xops
elif BACKEND == "flash_attn":
    import flash_attn
elif BACKEND == "sdpa":
    from torch.nn.functional import scaled_dot_product_attention as sdpa
elif BACKEND == "naive":
    pass
else:
    raise ValueError(f"Unknown attention backend: {BACKEND}")


__all__ = ["scaled_dot_product_attention", "modify_attn_score"]


def modify_attn_score(
    attn_score: torch.Tensor,
    lambda_scale: float = 0.3,
    max_passes: int = 4,
    stop_conflict: float = 0.5,
) -> torch.Tensor:
    """Reduce many-query-to-one-key conflicts in raw attention logits."""
    score = attn_score.float().clone()
    dtype = attn_score.dtype
    bsz, heads, query_len, key_len = score.shape
    flat = score.reshape(bsz * heads, query_len, key_len)

    for _ in range(max(int(max_passes), 0)):
        best_val, best_key = flat.max(dim=-1)
        counts = torch.zeros(flat.shape[0], key_len, device=flat.device, dtype=flat.dtype)
        counts.scatter_add_(1, best_key, torch.ones_like(best_val))
        overload = torch.relu(counts - 1.0)
        if float(overload.sum(dim=1).mean()) <= float(stop_conflict):
            break

        winner = torch.full_like(counts, -torch.inf)
        winner.scatter_reduce_(1, best_key, best_val, reduce="amax", include_self=True)
        winner_for_query = winner.gather(1, best_key)
        crowd_for_query = overload.gather(1, best_key)
        loser = (counts.gather(1, best_key) > 1) & (best_val < winner_for_query) & (best_val > 0)
        if not bool(loser.any()):
            break

        loser_idx = loser.nonzero(as_tuple=False)
        m_idx, q_idx = loser_idx[:, 0], loser_idx[:, 1]
        k_idx = best_key[m_idx, q_idx]
        cur = flat[m_idx, q_idx, k_idx]
        penalty = float(lambda_scale) * (1.0 + crowd_for_query[m_idx, q_idx]) * winner_for_query[m_idx, q_idx]
        flat[m_idx, q_idx, k_idx] = torch.clamp(cur - penalty, min=0.0)

    return flat.reshape(bsz, heads, query_len, key_len).to(dtype)


def _score_from_attention(attn_weight: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    score_per_head = attn_weight.max(dim=-1).values.permute(0, 2, 1)
    head_weight = torch.softmax(out.norm(dim=-1), dim=-1)
    return (score_per_head * head_weight).sum(dim=-1)


def _naive_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    return_score: bool = False,
    modify: bool = False,
    gate_attn: bool = False,
    modify_lambda_scale: float = 0.3,
) -> torch.Tensor:
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)
    logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if modify:
        logits = modify_attn_score(logits, lambda_scale=modify_lambda_scale)
    attn_weight = torch.softmax(logits, dim=-1)
    out = torch.matmul(attn_weight, v)

    if gate_attn:
        entropy = -(attn_weight * torch.log(attn_weight + 1e-8)).sum(dim=-1, keepdim=True)
        max_logits = logits.max(dim=-1, keepdim=True).values
        gate = ~((entropy > 6.0) & (max_logits < 1.0))
        out = out * gate.to(out.dtype)

    out = out.permute(0, 2, 1, 3)
    if return_score:
        return out, _score_from_attention(attn_weight, out)
    return out


@overload
def scaled_dot_product_attention(qkv: torch.Tensor) -> torch.Tensor: ...


@overload
def scaled_dot_product_attention(q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor: ...


@overload
def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor: ...


def scaled_dot_product_attention(*args, **kwargs):
    arg_names = {1: ["qkv"], 2: ["q", "kv"], 3: ["q", "k", "v"]}
    num_args = len(args)
    assert num_args in arg_names, f"Invalid number of arguments: {num_args}"

    return_score = bool(kwargs.get("return_score", False))
    modify = bool(kwargs.get("modify", False))
    gate_attn = bool(kwargs.get("gate_attn", False))
    modify_lambda_scale = float(kwargs.get("modify_lambda_scale", 0.3))

    if num_args == 1:
        qkv = args[0]
        assert qkv.ndim == 5 and qkv.shape[2] == 3, f"Expected [N, L, 3, H, C], got {tuple(qkv.shape)}"
        q, k, v = qkv.unbind(dim=2)
    elif num_args == 2:
        q, kv = args
        assert q.ndim == 4 and kv.ndim == 5 and kv.shape[2] == 2
        k, v = kv.unbind(dim=2)
    else:
        q, k, v = args
        assert q.ndim == k.ndim == v.ndim == 4

    if return_score or modify or gate_attn or BACKEND == "naive":
        return _naive_sdpa(
            q,
            k,
            v,
            return_score=return_score,
            modify=modify,
            gate_attn=gate_attn,
            modify_lambda_scale=modify_lambda_scale,
        )

    if BACKEND == "xformers":
        return xops.memory_efficient_attention(q, k, v)
    if BACKEND == "flash_attn":
        if num_args == 1:
            return flash_attn.flash_attn_qkvpacked_func(args[0])
        if num_args == 2:
            return flash_attn.flash_attn_kvpacked_func(q, args[1])
        return flash_attn.flash_attn_func(q, k, v)
    if BACKEND == "sdpa":
        out = sdpa(q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3))
        return out.permute(0, 2, 1, 3)
    raise ValueError(f"Unknown attention backend: {BACKEND}")
