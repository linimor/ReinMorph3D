from typing import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..attention import MultiHeadAttention
from ..norm import LayerNorm32
from .blocks import FeedForwardNet
from ...utils.morphing_utils import *
from ...utils.ot_coherence import (
    get_ot_filter_lambda,
    get_ss_token_positions,
    ot_motion_coherent_filter,
)

class ModulatedTransformerBlock(nn.Module):
    """
    Transformer block (MSA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        h = self.attn(h)
        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, use_reentrant=False)
        else:
            return self._forward(x, mod)


class ModulatedTransformerCrossBlock(nn.Module):
    """
    Transformer cross-attention block (MSA + MCA + FFN) with adaptive layer norm conditioning.
    """
    def __init__(
        self,
        channels: int,
        ctx_channels: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        attn_mode: Literal["full", "windowed"] = "full",
        window_size: Optional[int] = None,
        shift_window: Optional[Tuple[int, int, int]] = None,
        use_checkpoint: bool = False,
        use_rope: bool = False,
        qk_rms_norm: bool = False,
        qk_rms_norm_cross: bool = False,
        qkv_bias: bool = True,
        share_mod: bool = False,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = MultiHeadAttention(
            channels,
            num_heads=num_heads,
            type="self",
            attn_mode=attn_mode,
            window_size=window_size,
            shift_window=shift_window,
            qkv_bias=qkv_bias,
            use_rope=use_rope,
            qk_rms_norm=qk_rms_norm,
        )
        self.cross_attn = MultiHeadAttention(
            channels,
            ctx_channels=ctx_channels,
            num_heads=num_heads,
            type="cross",
            attn_mode="full",
            qkv_bias=qkv_bias,
            qk_rms_norm=qk_rms_norm_cross,
        )
        self.mlp = FeedForwardNet(
            channels,
            mlp_ratio=mlp_ratio,
        )
        if not share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(),
                nn.Linear(channels, 6 * channels, bias=True)
            )

    def _forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, step_idx: int, block_idx: int, **kwargs):
        if self.share_mod:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mod.chunk(6, dim=1)
        else:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(mod).chunk(6, dim=1)
        h = self.norm1(x)
        h = h * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)

        if len(kwargs) > 0:
            if kwargs["ss_tfsa_flag"]:
                h_cur, score_cur = self.self_attn(x=h, step_idx=step_idx, block_idx=block_idx, return_score=True, **kwargs)
                h_prev, score_prev = self.self_attn(x=h, step_idx=step_idx, block_idx=block_idx, cache_idx=-1, return_score=True, **kwargs)
                # score_diff = score_prev - score_cur
                # lambda_ = 5.0
                # delta_alpha = 0.3 * torch.tanh(lambda_ * score_diff)
                # delta_alpha = delta_alpha.unsqueeze(-1)  # [B, Lq, 1]
                # tfsa_alpha = torch.clamp(kwargs["tfsa_alpha"] - delta_alpha, 0.0, 1.0)
                h = feature_interp(h_cur, h_prev, kwargs["tfsa_alpha"], interp_mode="linear")
            else:
                h = self.self_attn(x=h, step_idx=step_idx, block_idx=block_idx, **kwargs)
        else:
            h = self.self_attn(x=h, step_idx=step_idx, block_idx=block_idx)

        h = h * gate_msa.unsqueeze(1)
        x = x + h
        h = self.norm2(x)

        if len(kwargs) > 0:
            if kwargs["ss_mca_flag"]:
                attn_kwargs = {
                    "modify": kwargs.get("modify", False),
                    "gate_attn": kwargs.get("gate_attn", False),
                    "modify_lambda_scale": kwargs.get("modify_lambda_scale", 0.3),
                }
                h_src = self.cross_attn(x=h, context=context, step_idx=step_idx, block_idx=block_idx, **attn_kwargs)
                h_tar = self.cross_attn(x=h, context=kwargs["tar_cond"], step_idx=step_idx, block_idx=block_idx, **attn_kwargs)
                # print(score_src.shape) #[1, 4096]
                # src_score = score_src          # [B, Lq]
                # tar_score = score_tar          # [B, Lq]
                # score_diff = tar_score - src_score   # [B, Lq]
                # lambda_ = 5.0
                # delta_alpha = 0.3 * torch.tanh(lambda_ * score_diff)   # [B, Lq]
                # delta_alpha = delta_alpha.unsqueeze(-1)  # [B, Lq, 1]
                # alpha = torch.clamp(kwargs["alpha"] - delta_alpha, 0.0, 1.0)
                h = feature_interp(h_src, h_tar, kwargs["alpha"], interp_mode="linear")
                if kwargs.get("ot_coherence_enabled", False) and kwargs.get("ot_coherence_stage", "ss") == "ss":
                    motion_field = kwargs.get("ot_motion_field", None)
                    if motion_field is not None:
                        try:
                            alpha = float(kwargs["alpha"])
                            src_center = motion_field["src_center"].to(device=h.device, dtype=h.dtype)
                            disp = motion_field["disp"].to(device=h.device, dtype=h.dtype)
                            anchor_pos = src_center + (1.0 - alpha) * disp
                            lam = get_ot_filter_lambda(
                                kwargs.get("ot_filter_lambda", 0.3),
                                step_idx,
                                kwargs.get("ss_num_steps", kwargs.get("steps", None)),
                                kwargs.get("ot_filter_start_step_ratio", 1.0),
                                kwargs.get("ot_filter_end_step_ratio", 0.0),
                            )
                            token_pos = get_ss_token_positions(
                                h,
                                ss_coords=kwargs.get("ss_coords", None),
                                grid_size=kwargs.get("ss_token_grid_size", None),
                            )
                            h = ot_motion_coherent_filter(
                                out=h,
                                token_pos=token_pos,
                                anchor_pos=anchor_pos,
                                anchor_motion=disp,
                                anchor_conf=motion_field.get("conf", None),
                                anchor_mask=motion_field.get("mask", None),
                                k_neighbors=kwargs.get("ot_filter_k", 16),
                                sigma_pos=kwargs.get("ot_filter_sigma_pos", 2.0),
                                sigma_motion=kwargs.get("ot_filter_sigma_motion", 2.0),
                                lambda0=lam,
                                use_confidence=kwargs.get("ot_filter_use_confidence", True),
                            )
                        except Exception as exc:
                            if kwargs.get("ot_debug", False):
                                print(f"[OT coherence] fallback to raw SS MCA residual: {exc}")
            else:
                h = self.cross_attn(x=h, context=context, step_idx=step_idx, block_idx=block_idx, **kwargs)
        else:
            h = self.cross_attn(x=h, context=context, step_idx=step_idx, block_idx=block_idx)

        x = x + h
        h = self.norm3(x)
        h = h * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        h = self.mlp(h)
        h = h * gate_mlp.unsqueeze(1)
        x = x + h
        return x

    def forward(self, x: torch.Tensor, mod: torch.Tensor, context: torch.Tensor, step_idx: int, block_idx: int, **kwargs):
        if self.use_checkpoint:
            return torch.utils.checkpoint.checkpoint(self._forward, x, mod, context, step_idx, block_idx, **kwargs, use_reentrant=False)
        else:
            return self._forward(x, mod, context, step_idx, block_idx, **kwargs)
        
