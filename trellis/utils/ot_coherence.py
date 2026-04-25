from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _batched_xyz(coords: torch.Tensor) -> torch.Tensor:
    if coords.ndim == 2:
        if coords.shape[-1] == 4:
            coords = coords[:, 1:]
        if coords.shape[-1] != 3:
            raise ValueError(f"Expected [N, 3] or [N, 4], got {tuple(coords.shape)}")
        return coords.unsqueeze(0)
    if coords.ndim == 3 and coords.shape[-1] == 3:
        return coords
    raise ValueError(f"Expected [N, 3] or [B, N, 3], got {tuple(coords.shape)}")


def _normalize_xyz(coords: torch.Tensor, grid_size: int = 64) -> torch.Tensor:
    coords = torch.nan_to_num(coords.float(), nan=0.0, posinf=float(grid_size - 1), neginf=0.0)
    if coords.numel() == 0:
        return coords
    if float(coords.min()) >= -1.0001 and float(coords.max()) <= 1.0001:
        return coords.clamp(-1.0, 1.0)
    denom = max(float(grid_size - 1), 1.0)
    return coords.clamp(0.0, denom).div(denom).mul(2.0).sub(1.0)


def _norm_to_grid(coords: torch.Tensor, grid_size: int) -> torch.Tensor:
    denom = max(float(grid_size - 1), 1.0)
    return ((coords.clamp(-1.0, 1.0) + 1.0) * 0.5 * denom).round().long().clamp(0, grid_size - 1)


def build_ss_anchors(
    active_voxels: torch.Tensor,
    grid_size: int = 64,
    patch_size: int = 4,
    max_anchors: int = 512,
) -> Dict[str, torch.Tensor]:
    points = _batched_xyz(active_voxels)
    device = points.device
    grid_size = int(grid_size)
    patch_size = int(patch_size)
    max_anchors = int(max_anchors)
    if grid_size <= 1 or patch_size <= 0 or max_anchors <= 0:
        raise ValueError("grid_size, patch_size, and max_anchors must be positive")

    all_centers, all_feats, all_mass = [], [], []
    patch_grid = (grid_size + patch_size - 1) // patch_size
    patch_volume = float(patch_size ** 3)

    for pts in points:
        pts = pts[torch.isfinite(pts).all(dim=-1)]
        if pts.numel() == 0:
            all_centers.append(torch.zeros(0, 3, device=device))
            all_feats.append(torch.zeros(0, 10, device=device))
            all_mass.append(torch.zeros(0, device=device))
            continue

        pts_norm = _normalize_xyz(pts, grid_size)
        pts_grid = _norm_to_grid(pts_norm, grid_size)
        patch_idx = torch.div(pts_grid, patch_size, rounding_mode="floor")
        patch_linear = patch_idx[:, 0] * patch_grid * patch_grid + patch_idx[:, 1] * patch_grid + patch_idx[:, 2]
        unique_patch, inverse = torch.unique(patch_linear, sorted=False, return_inverse=True)

        centers, feats, masses = [], [], []
        for local_idx, patch_id in enumerate(unique_patch):
            mask = inverse == local_idx
            cur_norm = pts_norm[mask]
            cur_grid = pts_grid[mask]
            count = float(mask.sum().item())
            center = cur_norm.mean(dim=0).clamp(-1.0, 1.0)

            patch_coord = torch.stack([
                patch_id // (patch_grid * patch_grid),
                (patch_id // patch_grid) % patch_grid,
                patch_id % patch_grid,
            ]).to(device=device, dtype=torch.float32)
            patch_center_grid = (patch_coord * patch_size + (patch_size - 1) * 0.5).clamp(0, grid_size - 1)
            patch_center = patch_center_grid.div(max(float(grid_size - 1), 1.0)).mul(2.0).sub(1.0)

            density = torch.tensor([min(count / patch_volume, 1.0)], device=device)
            offset = (center - patch_center) / max(2.0 * patch_size / max(float(grid_size - 1), 1.0), 1e-6)
            extent = (cur_grid.max(dim=0).values.float() - cur_grid.min(dim=0).values.float())
            extent = extent.div(max(float(patch_size - 1), 1.0)).clamp(0.0, 1.0)
            feat = torch.cat([density, offset.clamp(-1.0, 1.0), center, extent], dim=0)

            centers.append(center)
            feats.append(feat)
            masses.append(count)

        centers = torch.stack(centers)
        feats = torch.stack(feats)
        mass = torch.tensor(masses, device=device, dtype=torch.float32)
        if centers.shape[0] > max_anchors:
            keep = torch.topk(mass, max_anchors, largest=True).indices
            centers, feats, mass = centers[keep], feats[keep], mass[keep]
        mass = mass / mass.sum().clamp(min=1e-8)
        all_centers.append(centers)
        all_feats.append(feats)
        all_mass.append(mass)

    width = max([x.shape[0] for x in all_centers] + [1])

    def pad(values: list[torch.Tensor], trailing: tuple[int, ...]) -> torch.Tensor:
        out = []
        for value in values:
            if value.shape[0] < width:
                value = torch.cat([value, value.new_zeros(width - value.shape[0], *trailing)], dim=0)
            out.append(value)
        return torch.stack(out)

    mass = pad(all_mass, ())
    return {
        "center": pad(all_centers, (3,)),
        "feat": pad(all_feats, (10,)),
        "mass": mass,
        "mask": mass > 0,
    }


def _normalize_mass(mass: torch.Tensor, mask: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    mass = torch.nan_to_num(mass.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    valid = (mass > 0) if mask is None else mask.to(device=mass.device, dtype=torch.bool)
    mass = torch.where(valid, mass, torch.zeros_like(mass))
    uniform = valid.float() / valid.float().sum(dim=-1, keepdim=True).clamp(min=1.0)
    mass = torch.where(mass.sum(dim=-1, keepdim=True) > 0, mass / mass.sum(dim=-1, keepdim=True).clamp(min=1e-8), uniform)
    return mass, valid


def sinkhorn_transport(
    cost: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 0.05,
    n_iters: int = 80,
    src_mask: Optional[torch.Tensor] = None,
    tgt_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if cost.ndim != 3:
        raise ValueError(f"cost must be [B, Ks, Kt], got {tuple(cost.shape)}")
    eps = max(float(eps), 1e-8)
    a, src_valid = _normalize_mass(a, src_mask)
    b, tgt_valid = _normalize_mass(b, tgt_mask)
    valid_pair = src_valid.unsqueeze(-1) & tgt_valid.unsqueeze(-2)
    fallback = torch.where(valid_pair, a.unsqueeze(-1) * b.unsqueeze(-2), torch.zeros_like(cost.float()))

    safe_cost = torch.nan_to_num(cost.float(), nan=1e8, posinf=1e8, neginf=1e8)
    log_k = torch.where(valid_pair, -safe_cost / eps, torch.full_like(safe_cost, -torch.inf))
    log_a = torch.log(a.clamp(min=1e-8))
    log_b = torch.log(b.clamp(min=1e-8))
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)

    for _ in range(max(int(n_iters), 0)):
        log_u = log_a - torch.logsumexp(log_k + log_v.unsqueeze(-2), dim=-1)
        log_v = log_b - torch.logsumexp(log_k + log_u.unsqueeze(-1), dim=-2)
        log_u = torch.where(src_valid, log_u, torch.zeros_like(log_u))
        log_v = torch.where(tgt_valid, log_v, torch.zeros_like(log_v))

    plan = torch.exp(log_k + log_u.unsqueeze(-1) + log_v.unsqueeze(-2))
    plan = torch.where(valid_pair, plan, torch.zeros_like(plan))
    good = torch.isfinite(plan).flatten(1).all(dim=-1) & (plan.flatten(1).sum(dim=-1) > 0)
    return torch.where(good.view(-1, 1, 1), plan, fallback)


def build_ot_motion_field(src_anchors: Dict[str, torch.Tensor], tgt_anchors: Dict[str, torch.Tensor], cfg: Any) -> Dict[str, torch.Tensor]:
    src_center = src_anchors["center"].float()
    tgt_center = tgt_anchors["center"].to(src_center.device).float()
    src_feat = F.normalize(src_anchors["feat"].to(src_center.device).float(), dim=-1, eps=1e-6)
    tgt_feat = F.normalize(tgt_anchors["feat"].to(src_center.device).float(), dim=-1, eps=1e-6)
    src_mass = src_anchors["mass"].to(src_center.device).float()
    tgt_mass = tgt_anchors["mass"].to(src_center.device).float()
    src_mask = src_anchors.get("mask", src_mass > 0).to(src_center.device).bool()
    tgt_mask = tgt_anchors.get("mask", tgt_mass > 0).to(src_center.device).bool()

    c_pos = torch.cdist(src_center, tgt_center).square()
    c_feat = torch.cdist(src_feat, tgt_feat).square()
    cost = float(_cfg_get(cfg, "ot_cost_pos_weight", 0.3)) * c_pos + float(_cfg_get(cfg, "ot_cost_feat_weight", 1.0)) * c_feat
    cost = torch.where(src_mask.unsqueeze(-1) & tgt_mask.unsqueeze(-2), cost, torch.full_like(cost, 1e8))

    transport = sinkhorn_transport(
        cost,
        src_mass,
        tgt_mass,
        eps=float(_cfg_get(cfg, "ot_sinkhorn_eps", 0.05)),
        n_iters=int(_cfg_get(cfg, "ot_sinkhorn_iters", 80)),
        src_mask=src_mask,
        tgt_mask=tgt_mask,
    )
    row_mass = transport.sum(dim=-1, keepdim=True)
    row = transport / row_mass.clamp(min=1e-8)
    tgt_center_hat = torch.bmm(row, tgt_center)
    valid = src_mask & (row_mass.squeeze(-1) > 1e-8)
    disp = torch.where(valid.unsqueeze(-1), tgt_center_hat - src_center, torch.zeros_like(src_center))
    conf = torch.where(valid, row.max(dim=-1).values.clamp(0, 1), torch.zeros_like(src_mass))
    return {
        "src_center": src_center,
        "tgt_center_hat": torch.where(valid.unsqueeze(-1), tgt_center_hat, torch.zeros_like(tgt_center_hat)),
        "disp": disp,
        "conf": conf,
        "mass": torch.where(src_mask, src_mass, torch.zeros_like(src_mass)),
        "mask": src_mask,
        "T": transport,
    }


def get_ss_token_positions(hidden: torch.Tensor, ss_coords: Optional[torch.Tensor] = None, grid_size: Optional[int] = None) -> torch.Tensor:
    if hidden.ndim != 3:
        raise ValueError(f"hidden must be [B, N, C], got {tuple(hidden.shape)}")
    batch, token_count, _ = hidden.shape
    if ss_coords is not None:
        coords = _batched_xyz(ss_coords).to(hidden.device)
        if coords.shape[0] == 1 and batch > 1:
            coords = coords.expand(batch, -1, -1)
        if coords.shape[:2] != (batch, token_count):
            raise ValueError(f"ss_coords shape {tuple(coords.shape)} does not match hidden {tuple(hidden.shape)}")
        return _normalize_xyz(coords, int(grid_size or 64)).to(dtype=hidden.dtype)

    side = int(grid_size or round(token_count ** (1.0 / 3.0)))
    if side ** 3 != token_count:
        raise ValueError(f"Cannot infer SS grid from N={token_count}; pass grid_size or ss_coords")
    axes = [torch.arange(side, device=hidden.device) for _ in range(3)]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(1, token_count, 3).float()
    coords = coords.div(max(float(side - 1), 1.0)).mul(2.0).sub(1.0)
    return coords.expand(batch, -1, -1).to(dtype=hidden.dtype)


def get_ot_filter_lambda(base_lambda: float, step_idx: Optional[int], num_steps: Optional[int], start_ratio: float = 1.0, end_ratio: float = 0.0) -> float:
    base_lambda = max(float(base_lambda), 0.0)
    if step_idx is None or num_steps is None or int(num_steps) <= 1:
        return base_lambda
    progress = float(step_idx) / float(max(int(num_steps) - 1, 1))
    ratio = float(start_ratio) + (float(end_ratio) - float(start_ratio)) * progress
    return base_lambda * max(ratio, 0.0)


def ot_motion_coherent_filter(
    out: torch.Tensor,
    token_pos: torch.Tensor,
    anchor_pos: torch.Tensor,
    anchor_motion: torch.Tensor,
    anchor_conf: Optional[torch.Tensor] = None,
    anchor_mask: Optional[torch.Tensor] = None,
    k_neighbors: int = 16,
    sigma_pos: float = 2.0,
    sigma_motion: float = 2.0,
    lambda0: float = 0.3,
    use_confidence: bool = True,
    eps: float = 1e-6,
    chunk_size: int = 2048,
) -> torch.Tensor:
    if out.ndim != 3 or token_pos.ndim != 3 or anchor_pos.ndim != 3 or anchor_motion.shape != anchor_pos.shape:
        raise ValueError("Invalid OT coherence tensor shapes")
    batch, token_count, channels = out.shape
    if token_count == 0 or lambda0 <= 0:
        return out

    device, dtype = out.device, out.dtype
    token_pos = token_pos.to(device=device, dtype=torch.float32)
    anchor_pos = anchor_pos.to(device=device, dtype=torch.float32)
    anchor_motion = anchor_motion.to(device=device, dtype=torch.float32)
    anchor_conf = torch.ones(anchor_pos.shape[:2], device=device) if anchor_conf is None else anchor_conf.to(device=device, dtype=torch.float32)
    anchor_mask = (anchor_conf > 0) if anchor_mask is None else anchor_mask.to(device=device).bool()
    if not bool(anchor_mask.any()):
        return out

    k_neighbors = max(1, min(int(k_neighbors), token_count))
    sigma_pos = max(float(sigma_pos), eps)
    sigma_motion = max(float(sigma_motion), eps)
    lambda0 = min(max(float(lambda0), 0.0), 1.0)
    chunk_size = max(int(chunk_size), 1)

    clean_anchor_motion = torch.where(anchor_mask.unsqueeze(-1), anchor_motion, torch.zeros_like(anchor_motion))
    token_motion, token_conf = [], []
    for start in range(0, token_count, chunk_size):
        end = min(start + chunk_size, token_count)
        logits = -torch.cdist(token_pos[:, start:end], anchor_pos).square() / (2.0 * sigma_pos * sigma_pos)
        logits = torch.where(anchor_mask.unsqueeze(1), logits, torch.full_like(logits, -1e8))
        weights = torch.softmax(logits, dim=-1)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp(min=eps)
        token_motion.append(torch.bmm(weights, clean_anchor_motion))
        token_conf.append((weights * anchor_conf.unsqueeze(1)).sum(dim=-1))
    token_motion = torch.cat(token_motion, dim=1)
    token_conf = torch.cat(token_conf, dim=1).clamp(0.0, 1.0)

    chunks = []
    for start in range(0, token_count, chunk_size):
        end = min(start + chunk_size, token_count)
        dist = torch.cdist(token_pos[:, start:end], token_pos).square()
        knn_dist, knn_idx = torch.topk(dist, k=k_neighbors, dim=-1, largest=False)
        neighbor_out = torch.gather(out.unsqueeze(1).expand(-1, end - start, -1, -1), 2, knn_idx.unsqueeze(-1).expand(-1, -1, -1, channels))
        neighbor_motion = torch.gather(token_motion.unsqueeze(1).expand(-1, end - start, -1, -1), 2, knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3))
        motion_dist = (neighbor_motion - token_motion[:, start:end].unsqueeze(-2)).square().sum(dim=-1)
        logits = -knn_dist / (2.0 * sigma_pos * sigma_pos) - motion_dist / (2.0 * sigma_motion * sigma_motion)
        weights = torch.softmax(logits, dim=-1).to(dtype)
        smooth = (weights.unsqueeze(-1) * neighbor_out).sum(dim=-2)
        if use_confidence:
            lam = lambda0 * (1.0 - token_conf[:, start:end])
        else:
            lam = torch.full((batch, end - start), lambda0, device=device)
        lam = lam.clamp(0.0, lambda0).unsqueeze(-1).to(dtype)
        chunks.append((1.0 - lam) * out[:, start:end] + lam * smooth)
    return torch.cat(chunks, dim=1)
