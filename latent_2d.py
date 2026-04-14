from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from sklearn.decomposition import PCA
except Exception:
    PCA = None


# =========================================================
# preview
# =========================================================
def save_ss_preview_png(voxels, preview_root, name, morphing_idx, cand_idx=None):
    import imageio
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from skimage import measure

    preview_dir = os.path.join(preview_root, f"preview_{name}")
    os.makedirs(preview_dir, exist_ok=True)

    v = voxels.detach().float().cpu()
    if v.ndim == 5:
        v = v[0, 0].numpy()
    elif v.ndim == 4:
        v = v[0].numpy()
    else:
        raise ValueError(f"Unexpected voxels shape: {tuple(voxels.shape)}")

    v = (v > 0).astype(np.uint8)

    if v.sum() == 0:
        img = np.zeros((256, 1024, 3), dtype=np.uint8)
        fname = f"frame_{morphing_idx:03d}.png" if cand_idx is None else f"frame_{morphing_idx:03d}_cand_{cand_idx:02d}.png"
        preview_path = os.path.join(preview_dir, fname)
        imageio.imwrite(preview_path, img)
        print(f"[Preview saved] {preview_path}")
        return preview_path

    verts, faces, normals, values = measure.marching_cubes(v, level=0.5)
    views = [(20, 35), (20, 125), (20, 215), (70, 35)]

    rendered = []
    for elev, azim in views:
        fig = plt.figure(figsize=(3, 3), dpi=128)
        ax = fig.add_subplot(111, projection="3d")

        mesh = Poly3DCollection(verts[faces], alpha=1.0)
        mesh.set_facecolor((0.7, 0.7, 0.85))
        mesh.set_edgecolor((0.2, 0.2, 0.2))
        mesh.set_linewidth(0.05)
        ax.add_collection3d(mesh)

        x_min, y_min, z_min = verts.min(axis=0)
        x_max, y_max, z_max = verts.max(axis=0)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(z_min, z_max)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect([x_max - x_min + 1e-6, y_max - y_min + 1e-6, z_max - z_min + 1e-6])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        ax.set_axis_off()

        fig.tight_layout(pad=0)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())
        rendered.append(buf[..., :3].copy())
        plt.close(fig)

    strip = np.concatenate(rendered, axis=1)
    fname = f"frame_{morphing_idx:03d}.png" if cand_idx is None else f"frame_{morphing_idx:03d}_cand_{cand_idx:02d}.png"
    preview_path = os.path.join(preview_dir, fname)
    imageio.imwrite(preview_path, strip)
    print(f"[Preview saved immediately] {preview_path}")
    return preview_path


# =========================================================
# utils
# =========================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_tensor(x: torch.Tensor) -> torch.Tensor:
    return x.detach().reshape(-1)


def tensor_sha1(x: torch.Tensor) -> str:
    y = x.detach().float().cpu().contiguous().numpy()
    return hashlib.sha1(y.tobytes()).hexdigest()[:16]


def to_device(x: torch.Tensor, device: str) -> torch.Tensor:
    return x.cpu() if device == "cpu" else x.to(device)


def score_to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


# =========================================================
# data
# =========================================================
@dataclass
class Cell:
    cell_id: str
    level: int
    parent_id: Optional[str]
    low_vals: List[float]
    high_vals: List[float]
    status: str = "pending"   # pending / dead / kept / split / terminal
    point_ids: Optional[List[str]] = None
    children_ids: Optional[List[str]] = None
    summary: Optional[Dict[str, Any]] = None


# =========================================================
# pipeline
# =========================================================
class UserAdapter:
    def __init__(self, device: str = "cuda", local_model_path: str = "./TRELLIS-image-large") -> None:
        self.device = device
        try:
            from trellis.pipelines import TrellisImageTo3DPipeline
        except Exception as e:
            raise RuntimeError(f"导入 TrellisImageTo3DPipeline 失败: {e}")

        self.pipeline = TrellisImageTo3DPipeline.from_pretrained(local_model_path)
        if self.pipeline is None:
            raise RuntimeError(f"from_pretrained 返回了 None，模型路径可能有问题: {local_model_path}")

        if self.device != "cpu":
            try:
                if hasattr(self.pipeline, "cuda"):
                    self.pipeline.cuda()
                elif hasattr(self.pipeline, "to"):
                    self.pipeline.to(self.device)
            except Exception as e:
                raise RuntimeError(f"pipeline 移动到设备 {self.device} 失败: {e}")

        print(f"[INFO] pipeline loaded: {type(self.pipeline)}")

    def decode_ss(self, zs: torch.Tensor):
        if self.pipeline is None:
            raise RuntimeError("pipeline 是 None，请检查 from_pretrained 或 .cuda() 调用")
        if not hasattr(self.pipeline, "decode_ss_latent"):
            raise AttributeError(f"当前 pipeline 没有 decode_ss_latent，type={type(self.pipeline)}")

        if zs.ndim == 6 and zs.shape[0] == 1:
            zs = zs.squeeze(0)
        if zs.ndim != 5:
            raise ValueError(f"decode_ss_latent 期望 5D 输入 [B,C,D,H,W]，当前是 {tuple(zs.shape)}")

        with torch.no_grad():
            out = self.pipeline.decode_ss_latent(zs)

        if isinstance(out, dict):
            voxels = out.get("voxels", None)
            coords = out.get("coords", None)
            if voxels is None or coords is None:
                raise ValueError("decode_ss_latent 返回 dict 时，必须至少包含 voxels 和 coords")
            return voxels, coords

        if isinstance(out, (tuple, list)) and len(out) >= 2:
            return out[0], out[1]

        raise ValueError("decode_ss_latent 返回格式无法识别")

    def render_multiview(self, voxels: Any, save_dir: Path, point_meta: Dict[str, Any]) -> None:
        ensure_dir(save_dir)
        point_dir = save_dir.parent
        point_id = str(point_meta.get("point_id", "point"))
        level = int(point_meta.get("level", 0))
        cand_idx = point_meta.get("cand_idx", None)
        png_path = save_ss_preview_png(voxels, str(point_dir), point_id, level, cand_idx)
        save_json(save_dir / "render_meta.json", {"preview_png": str(png_path)})


# =========================================================
# space
# =========================================================
class LatentSearchSpace:
    """
    z(t, a1, a2, ...) = (1-t) * src + t * tar + w(t) * sum_i ai * basis_i
    t in [0, 1], w(t)=4t(1-t)
    也就是 src-tar 之间的内部空间，而不是围绕 mid 的盒子。
    """

    def __init__(
        self,
        zs_a: torch.Tensor,
        zs_b: torch.Tensor,
        side_radius: float = 0.35,
        active_basis_count: int = 6,
        block_divs: Tuple[int, int, int] = (2, 4, 4),
    ) -> None:
        self.src = zs_a.detach().clone().float().cpu()
        self.tar = zs_b.detach().clone().float().cpu()
        if self.src.shape != self.tar.shape:
            raise ValueError(f"两个 zs 形状不一致: {self.src.shape} vs {self.tar.shape}")
        if self.src.ndim != 5:
            raise ValueError(f"当前脚本默认 zs 是 5D [B,C,D,H,W]，现在拿到 {tuple(self.src.shape)}")

        self.shape = tuple(self.src.shape)
        self.delta = self.tar - self.src
        self.side_radius = float(side_radius)

        _, C, D, H, W = self.shape
        dz, dy, dx = block_divs
        z_edges = np.linspace(0, D, dz + 1, dtype=int)
        y_edges = np.linspace(0, H, dy + 1, dtype=int)
        x_edges = np.linspace(0, W, dx + 1, dtype=int)

        blocks: List[torch.Tensor] = []
        block_scores: List[float] = []
        abs_delta = self.delta.abs()

        for iz in range(dz):
            for iy in range(dy):
                for ix in range(dx):
                    z0, z1 = z_edges[iz], z_edges[iz + 1]
                    y0, y1 = y_edges[iy], y_edges[iy + 1]
                    x0, x1 = x_edges[ix], x_edges[ix + 1]

                    mask = torch.zeros_like(self.delta)
                    mask[:, :, z0:z1, y0:y1, x0:x1] = 1.0
                    basis = self.delta * mask
                    score = float(abs_delta[:, :, z0:z1, y0:y1, x0:x1].mean().item())
                    blocks.append(basis)
                    block_scores.append(score)

        order = np.argsort(block_scores)[::-1]
        k = min(active_basis_count, len(order))
        self.basis_ids = [int(i) for i in order[:k]]
        self.bases = [blocks[i].clone() for i in self.basis_ids]
        self.basis_scores = [block_scores[i] for i in self.basis_ids]

        self.root_low = [0.0] + [-self.side_radius] * len(self.bases)
        self.root_high = [1.0] + [self.side_radius] * len(self.bases)

    def make_root_cell(self) -> Cell:
        return Cell(
            cell_id="L0_C0000",
            level=0,
            parent_id=None,
            low_vals=list(self.root_low),
            high_vals=list(self.root_high),
        )

    def point_from_coeffs(self, coeffs: Sequence[float]) -> torch.Tensor:
        t = float(coeffs[0])
        t = max(0.0, min(1.0, t))
        z = (1.0 - t) * self.src + t * self.tar
        w = 4.0 * t * (1.0 - t)
        for a, basis in zip(coeffs[1:], self.bases):
            z = z + w * float(a) * basis
        return z.clone()

    def split_cell(self, cell: Cell) -> List[Cell]:
        mids = [(lo + hi) / 2.0 for lo, hi in zip(cell.low_vals, cell.high_vals)]
        children: List[Cell] = []
        for child_i, bits in enumerate(product([0, 1], repeat=len(cell.low_vals))):
            low, high = [], []
            for bit, lo, hi, md in zip(bits, cell.low_vals, cell.high_vals, mids):
                if bit == 0:
                    low.append(lo)
                    high.append(md)
                else:
                    low.append(md)
                    high.append(hi)
            cid = f"L{cell.level + 1}_C{child_i:04d}_{cell.cell_id}"
            children.append(Cell(cell_id=cid, level=cell.level + 1, parent_id=cell.cell_id, low_vals=low, high_vals=high))
        return children

    def sample_points_in_cell(self, cell: Cell, n_samples: int, n_candidates: int, seed: int = 42) -> List[List[float]]:
        rng = np.random.default_rng(seed + cell.level * 10007 + abs(hash(cell.cell_id)) % 100000)

        dim = len(cell.low_vals)
        lows = np.array(cell.low_vals, dtype=np.float32)
        highs = np.array(cell.high_vals, dtype=np.float32)
        candidates = rng.uniform(lows[None, :], highs[None, :], size=(n_candidates, dim))

        special = []
        center = ((lows + highs) / 2.0).astype(np.float32)
        special.append(center.copy())

        for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
            if t < lows[0] or t > highs[0]:
                continue
            p = center.copy()
            p[0] = t
            special.append(p)

        for t in [0.25, 0.5, 0.75]:
            if t < lows[0] or t > highs[0]:
                continue
            for i in range(1, dim):
                p1 = center.copy()
                p2 = center.copy()
                p1[0] = t
                p2[0] = t
                p1[i] = lows[i]
                p2[i] = highs[i]
                special.append(p1)
                special.append(p2)

        if len(special) > 0:
            candidates = np.concatenate([np.stack(special, axis=0), candidates], axis=0)

        chosen = []
        used = np.zeros(len(candidates), dtype=bool)
        if len(candidates) == 0:
            return []

        first_idx = 0
        chosen.append(candidates[first_idx])
        used[first_idx] = True

        while len(chosen) < n_samples and (~used).sum() > 0:
            chosen_arr = np.stack(chosen, axis=0)
            remain_idx = np.where(~used)[0]
            remain = candidates[remain_idx]
            d2 = ((remain[:, None, :] - chosen_arr[None, :, :]) ** 2).sum(axis=2)
            min_d2 = d2.min(axis=1)
            best_local = int(np.argmax(min_d2))
            best_idx = int(remain_idx[best_local])
            chosen.append(candidates[best_idx])
            used[best_idx] = True

        uniq = []
        seen = set()
        for p in chosen:
            key = tuple(np.round(p, 6).tolist())
            if key not in seen:
                uniq.append(p.tolist())
                seen.add(key)
        return uniq


# =========================================================
# db
# =========================================================
class SearchDB:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.points_dir = root / "points"
        self.maps_dir = root / "maps"
        self.db_json = root / "points_db.json"
        self.cells_json = root / "cells_db.json"
        self.run_json = root / "run_state.json"
        self.scores_csv = root / "scores_template.csv"
        self.summary_csv = root / "points_summary.csv"
        ensure_dir(self.points_dir)
        ensure_dir(self.maps_dir)

        self.points_db: Dict[str, Dict[str, Any]] = load_json(self.db_json, {})
        self.cells_db: Dict[str, Dict[str, Any]] = load_json(self.cells_json, {})
        self.run_state: Dict[str, Any] = load_json(self.run_json, {})

    def save(self) -> None:
        save_json(self.db_json, self.points_db)
        save_json(self.cells_json, self.cells_db)
        save_json(self.run_json, self.run_state)
        self._save_points_summary()

    def register_cell(self, cell: Cell) -> None:
        self.cells_db[cell.cell_id] = asdict(cell)

    def update_cell(self, cell: Cell) -> None:
        self.cells_db[cell.cell_id] = asdict(cell)

    def has_point_hash(self, point_hash: str) -> Optional[str]:
        for pid, meta in self.points_db.items():
            if meta.get("point_hash") == point_hash:
                return pid
        return None

    def register_point(self, point_id: str, level: int, cell_id: str, zs: torch.Tensor, coeffs: Sequence[float], point_hash: str) -> Path:
        pdir = self.points_dir / point_id
        ensure_dir(pdir)
        self.points_db[point_id] = {
            "point_id": point_id,
            "level": int(level),
            "cell_id": cell_id,
            "point_hash": point_hash,
            "coeffs": [float(v) for v in coeffs],
            "point_dir": str(pdir),
            "zs_path": str(pdir / "zs.pt"),
            "voxels_path": str(pdir / "voxels.pt"),
            "coords_path": str(pdir / "coords.pt"),
            "score": None,
            "status": "generated",
        }
        torch.save(zs.detach().cpu(), pdir / "zs.pt")
        return pdir

    def set_point_eval_done(self, point_id: str) -> None:
        self.points_db[point_id]["status"] = "evaluated"

    def set_point_score(self, point_id: str, score: float) -> None:
        self.points_db[point_id]["score"] = float(score)
        self.points_db[point_id]["status"] = "scored"

    def iter_scored_points(self) -> Iterable[Tuple[str, Dict[str, Any]]]:
        for pid, meta in self.points_db.items():
            if score_to_float(meta.get("score")) is not None:
                yield pid, meta

    def _save_points_summary(self) -> None:
        rows = []
        for pid, meta in self.points_db.items():
            rows.append({
                "point_id": pid,
                "level": meta.get("level"),
                "cell_id": meta.get("cell_id"),
                "score": meta.get("score"),
                "status": meta.get("status"),
                "point_dir": meta.get("point_dir"),
                "zs_path": meta.get("zs_path"),
            })
        with open(self.summary_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["point_id", "level", "cell_id", "score", "status", "point_dir", "zs_path"])
            writer.writeheader()
            writer.writerows(rows)

    def export_score_template(self) -> None:
        rows = []
        for pid, meta in sorted(self.points_db.items(), key=lambda kv: kv[0]):
            rows.append({
                "point_id": pid,
                "score": "" if meta.get("score") is None else meta.get("score"),
                "status": meta.get("status", ""),
                "level": meta.get("level", ""),
                "cell_id": meta.get("cell_id", ""),
                "point_dir": meta.get("point_dir", ""),
            })
        with open(self.scores_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["point_id", "score", "status", "level", "cell_id", "point_dir"]
            )
            writer.writeheader()
            writer.writerows(rows)

    def import_scores(self, score_csv: Optional[Path] = None) -> int:
        csv_path = score_csv or self.scores_csv
        print(f"[DEBUG] import_scores csv_path = {csv_path}")

        if not csv_path.exists():
            print(f"[WARN] 评分表不存在: {csv_path}")
            return 0

        updated = 0
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("point_id", "").strip()
                if pid not in self.points_db:
                    continue
                sc = score_to_float(row.get("score", None))
                if sc is None:
                    continue
                self.set_point_score(pid, sc)
                updated += 1

        return updated


# =========================================================
# map
# =========================================================
class MapBuilder:
    def __init__(self, db: SearchDB) -> None:
        self.db = db

    def build_pca_map(self, include_unscored: bool = True) -> Optional[Path]:
        if PCA is None:
            print("[WARN] sklearn 不可用，跳过 PCA 地图")
            return None

        pts: List[np.ndarray] = []
        pids: List[str] = []
        levels: List[int] = []
        scores: List[Optional[float]] = []

        for pid, meta in self.db.points_db.items():
            sc = score_to_float(meta.get("score"))
            if (not include_unscored) and sc is None:
                continue
            zs_path = Path(meta["zs_path"])
            if not zs_path.exists():
                continue
            z = torch.load(zs_path, map_location="cpu")
            pts.append(flatten_tensor(z).numpy())
            pids.append(pid)
            levels.append(int(meta.get("level", 0)))
            scores.append(sc)

        if len(pts) < 2:
            print("[WARN] 点少于 2 个，暂不生成 PCA 地图")
            return None

        X = np.stack(pts, axis=0)
        pca = PCA(n_components=2, random_state=0)
        XY = pca.fit_transform(X)
        if XY[:, 0].mean() < 0:
            XY[:, 0] *= -1.0
        if XY[:, 1].mean() < 0:
            XY[:, 1] *= -1.0

        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111)

        scored_mask = np.array([s is not None for s in scores], dtype=bool)
        if scored_mask.any():
            scored_idx = np.where(scored_mask)[0]
            sc = ax.scatter(XY[scored_idx, 0], XY[scored_idx, 1], c=np.array([scores[i] for i in scored_idx], dtype=float), s=80, cmap="viridis")
            plt.colorbar(sc, ax=ax, label="score")
        if (~scored_mask).any():
            uns_idx = np.where(~scored_mask)[0]
            ax.scatter(XY[uns_idx, 0], XY[uns_idx, 1], s=60, c="lightgray", marker="x", label="unscored")
            ax.legend()

        for x, y, pid, lvl, scv in zip(XY[:, 0], XY[:, 1], pids, levels, scores):
            if scv is None:
                label = f"{pid}\nL{lvl}|NA"
            else:
                label = f"{pid}\nL{lvl}|{scv:.1f}"
            ax.text(x, y, label, fontsize=7)

        ax.set_title("Latent map (PCA)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        plt.tight_layout()

        out = self.db.maps_dir / "latent_map_pca.png"
        plt.savefig(out, dpi=220)
        plt.close(fig)

        csv_path = self.db.maps_dir / "latent_map_pca.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["point_id", "x", "y", "score", "level"])
            writer.writeheader()
            for x, y, pid, scv, lvl in zip(XY[:, 0], XY[:, 1], pids, scores, levels):
                writer.writerow({"point_id": pid, "x": float(x), "y": float(y), "score": "" if scv is None else float(scv), "level": int(lvl)})

        return out


# =========================================================
# explorer
# =========================================================
class LatentGridExplorer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.out_dir = Path(args.out)
        ensure_dir(self.out_dir)

        if args.reset_state:
            for name in ["points", "maps", "points_db.json", "cells_db.json", "run_state.json", "scores_template.csv", "points_summary.csv"]:
                p = self.out_dir / name
                if p.is_dir():
                    shutil.rmtree(p, ignore_errors=True)
                elif p.exists():
                    p.unlink()

        self.db = SearchDB(self.out_dir)
        self.zs_a = torch.load(args.src, map_location="cpu").float()
        self.zs_b = torch.load(args.tar, map_location="cpu").float()

        self.space = LatentSearchSpace(
            self.zs_a,
            self.zs_b,
            side_radius=args.side_radius,
            active_basis_count=args.active_basis_count,
            block_divs=(args.block_div_z, args.block_div_y, args.block_div_x),
        )

        self.adapter = UserAdapter(device=args.device, local_model_path=args.local_model_path) if args.evaluate else None
        self._init_run_state()

    def _init_run_state(self) -> None:
        if not self.db.run_state:
            self.db.run_state = {
                "src": str(self.args.src),
                "tar": str(self.args.tar),
                "shape": list(self.zs_a.shape),
                "side_radius": float(self.args.side_radius),
                "active_basis_count": int(self.args.active_basis_count),
                "block_divs": [int(self.args.block_div_z), int(self.args.block_div_y), int(self.args.block_div_x)],
                "sample_per_cell": int(self.args.sample_per_cell),
                "candidate_per_cell": int(self.args.candidate_per_cell),
                "dead_threshold": float(self.args.dead_threshold),
                "keep_threshold": float(self.args.keep_threshold),
                "good_threshold": float(self.args.good_threshold),
                "min_refine_ratio": float(self.args.min_refine_ratio),
                "max_levels": int(self.args.max_levels),
                "basis_scores": list(self.space.basis_scores),
                "basis_ids": list(self.space.basis_ids),
            }
            self.db.save()

    def _load_cell(self, cell_id: str) -> Cell:
        return Cell(**self.db.cells_db[cell_id])

    def bootstrap_root(self) -> None:
        if self.db.cells_db:
            return
        root = self.space.make_root_cell()
        self.db.register_cell(root)
        self.db.save()

    def generate_points_for_cell(self, cell: Cell) -> List[str]:
        point_counter = len(self.db.points_db)
        coeff_list = self.space.sample_points_in_cell(cell, self.args.sample_per_cell, self.args.candidate_per_cell, self.args.seed)

        point_ids: List[str] = []
        kept_ids: List[str] = []
        src_skip_thresh = self.args.skip_near_src
        tar_skip_thresh = self.args.skip_near_tar

        for coeffs in coeff_list:
            zs = self.space.point_from_coeffs(coeffs)
            dist_src = (zs - self.zs_a).abs().mean().item()
            dist_tar = (zs - self.zs_b).abs().mean().item()

            if src_skip_thresh > 0 and dist_src < src_skip_thresh:
                print(f"[SKIP-SRC] coeffs={[round(float(x), 4) for x in coeffs]} dist_src={dist_src:.6f} dist_tar={dist_tar:.6f}")
                continue
            if tar_skip_thresh > 0 and dist_tar < tar_skip_thresh:
                print(f"[SKIP-TAR] coeffs={[round(float(x), 4) for x in coeffs]} dist_src={dist_src:.6f} dist_tar={dist_tar:.6f}")
                continue

            ph = tensor_sha1(zs)
            existed = self.db.has_point_hash(ph)
            if existed is not None:
                point_ids.append(existed)
                kept_ids.append(existed)
                continue

            pid = f"P{point_counter:06d}_L{cell.level}"
            point_counter += 1
            pdir = self.db.register_point(pid, cell.level, cell.cell_id, zs, coeffs, ph)
            point_meta = self.db.points_db[pid]
            save_json(pdir / "meta.json", point_meta)
            point_ids.append(pid)
            kept_ids.append(pid)

            if self.adapter is not None:
                try:
                    zs_in = to_device(zs, self.args.device)
                    voxels, coords = self.adapter.decode_ss(zs_in)
                    torch.save(voxels, pdir / "voxels.pt")
                    torch.save(coords, pdir / "coords.pt")
                    render_dir = pdir / "renders"
                    self.adapter.render_multiview(voxels, render_dir, point_meta)
                    self.db.set_point_eval_done(pid)
                except Exception as e:
                    with open(pdir / "decode_or_render_error.txt", "w", encoding="utf-8") as f:
                        f.write(str(e))
                    self.db.points_db[pid]["status"] = "error"

            print(
                f"[GEN] {pid} coeffs={[round(float(x), 4) for x in coeffs]} "
                f"dist_src={dist_src:.6f} dist_tar={dist_tar:.6f} hash={ph}"
            )

        cell.point_ids = kept_ids
        self.db.update_cell(cell)
        self.db.save()
        return point_ids

    def run_generate(self) -> None:
        self.bootstrap_root()

        generated_any = False
        for cell_id, raw in list(self.db.cells_db.items()):
            cell = Cell(**raw)
            if not cell.point_ids:
                self.generate_points_for_cell(cell)
                generated_any = True

        self.db.export_score_template()
        self.db.save()

        if generated_any:
            print(f"[INFO] 新点已生成，评分表更新到: {self.db.scores_csv}")
        else:
            print(f"[INFO] 没有新的待生成点，评分表已同步: {self.db.scores_csv}")

    def evaluate_cells_by_scores(self) -> Tuple[List[str], List[str], List[str]]:
        dead_cells: List[str] = []
        keep_cells: List[str] = []
        split_cells: List[str] = []

        for cell_id, raw in list(self.db.cells_db.items()):
            cell = Cell(**raw)
            if not cell.point_ids:
                continue

            scores = []
            for pid in cell.point_ids:
                if pid not in self.db.points_db:
                    continue
                sc = score_to_float(self.db.points_db[pid].get("score"))
                if sc is not None:
                    scores.append(sc)

            # 这个 cell 里的点评分没打全，就先不动
            if len(scores) < len(cell.point_ids):
                continue

            mx = max(scores)
            mn = min(scores)
            mean = float(np.mean(scores))
            std = float(np.std(scores))

            cell.summary = {
                "max_score": mx,
                "min_score": mn,
                "mean_score": mean,
                "std_score": std,
                "n_points": len(scores),
            }

            # 1) 死区
            if mx < self.args.dead_threshold:
                cell.status = "dead"
                dead_cells.append(cell.cell_id)

            # 2) 到最大层数了，不再细分
            elif cell.level >= self.args.max_levels:
                cell.status = "terminal"
                keep_cells.append(cell.cell_id)

            # 3) 除死区外，全部细分
            else:
                cell.status = "kept"
                split_cells.append(cell.cell_id)

            self.db.update_cell(cell)

        self.db.save()
        return dead_cells, keep_cells, split_cells

    def refine_once(self) -> int:
        _, _, split_cells = self.evaluate_cells_by_scores()
        new_children = 0
        for cid in split_cells:
            cell = self._load_cell(cid)
            if cell.children_ids:
                continue

            ratios = []
            for i, (lo, hi) in enumerate(zip(cell.low_vals, cell.high_vals)):
                init_w = 1.0 if i == 0 else 2.0 * self.args.side_radius
                cur_w = float(hi - lo)
                ratios.append(cur_w / max(init_w, 1e-12))

            if max(ratios) <= self.args.min_refine_ratio:
                cell.status = "terminal"
                self.db.update_cell(cell)
                continue

            children = self.space.split_cell(cell)
            child_ids = []
            for ch in children:
                self.db.register_cell(ch)
                self.generate_points_for_cell(ch)
                child_ids.append(ch.cell_id)
                new_children += 1

            cell.children_ids = child_ids
            cell.status = "split"
            self.db.update_cell(cell)

        self.db.export_score_template()
        self.db.save()
        return new_children

    def stop_reason(self) -> Optional[str]:
        scored = list(self.db.iter_scored_points())
        if len(scored) >= self.args.max_total_points:
            return f"已达到 max_total_points={self.args.max_total_points}"
        candidate_exists = False
        for _, raw in self.db.cells_db.items():
            cell = Cell(**raw)
            if cell.status in {"pending", "kept"}:
                candidate_exists = True
                break
        if not candidate_exists:
            return "没有可继续探索的 cell"
        return None

    def run_all(self) -> None:
        self.bootstrap_root()

        # 只画地图
        if self.args.map_only:
            if self.args.import_scores:
                n = self.db.import_scores(Path(self.args.score_csv) if self.args.score_csv else self.db.scores_csv)
                print(f"[INFO] 导入评分完成，更新 {n} 个点")
                self.db.save()

            mp = MapBuilder(self.db).build_pca_map(include_unscored=True)
            if mp is not None:
                print(f"[INFO] PCA 地图已保存: {mp}")
            return

        # 先导分
        if self.args.import_scores:
            n = self.db.import_scores(Path(self.args.score_csv) if self.args.score_csv else self.db.scores_csv)
            print(f"[INFO] 导入评分完成，更新 {n} 个点")
            self.db.save()

        # refine 优先
        if self.args.refine:
            new_children = self.refine_once()
            print(f"[INFO] refine 完成，新建 children cell 数: {new_children}")
            self.db.export_score_template()
            self.db.save()
            print(f"[INFO] 最新评分表: {self.db.scores_csv}")
        else:
            self.run_generate()

        # 最后单独画地图
        if self.args.import_scores or self.args.show_map_after_run:
            mp = MapBuilder(self.db).build_pca_map(include_unscored=True)
            if mp is not None:
                print(f"[INFO] PCA 地图已保存: {mp}")

        rs = self.stop_reason()
        if rs:
            print(f"[STOP] {rs}")


# =========================================================
# cli
# =========================================================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="src-tar 内部空间均匀采样 + 人工评分驱动细化")

    p.add_argument("--src", type=str, default="/root/autodl-tmp/MorphAny3D/outputs/cache/0001/cache/coords_zs.pt", help="起点 coords_zs.pt")
    p.add_argument("--tar", type=str, default="/root/autodl-tmp/MorphAny3D/outputs/cache/bee/cache/coords_zs.pt", help="终点 coords_zs.pt")
    p.add_argument("--out", type=str, default="/root/autodl-tmp/MorphAny3D/output", help="输出目录")

    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="decode 设备")
    p.add_argument("--local-model-path", type=str, default="./TRELLIS-image-large", help="本地模型目录")

    p.add_argument("--side-radius", type=float, default=0.35, help="侧向扰动半径")
    p.add_argument("--active-basis-count", type=int, default=6, help="使用 top-K coarse block basis")
    p.add_argument("--block-div-z", type=int, default=2, help="D 方向 coarse block 划分")
    p.add_argument("--block-div-y", type=int, default=4, help="H 方向 coarse block 划分")
    p.add_argument("--block-div-x", type=int, default=4, help="W 方向 coarse block 划分")

    p.add_argument("--sample-per-cell", type=int, default=12, help="每个 cell 均匀铺点数量")
    p.add_argument("--candidate-per-cell", type=int, default=256, help="每个 cell farthest-first 候选点数量")
    p.add_argument("--seed", type=int, default=42, help="随机种子")

    p.add_argument("--dead-threshold", type=float, default=3.0, help="cell 最高分低于此值则死区")
    p.add_argument("--keep-threshold", type=float, default=5.0, help="cell 均分达到此值视作可保留")
    p.add_argument("--good-threshold", type=float, default=7.0, help="cell 最高分达到此值强制细分")
    p.add_argument("--std-split-threshold", type=float, default=1.5, help="高方差细分阈值")
    p.add_argument("--min-refine-ratio", type=float, default=1/16, help="cell 最小细化比例")
    p.add_argument("--max-levels", type=int, default=4, help="最大细化层数")
    p.add_argument("--max-total-points", type=int, default=200, help="总点数预算")

    p.add_argument("--skip-near-src", type=float, default=0.0, help="若与 src 的 mean abs distance 小于该阈值则跳过")
    p.add_argument("--skip-near-tar", type=float, default=0.0, help="若与 tar 的 mean abs distance 小于该阈值则跳过")

    p.add_argument("--evaluate", action="store_true", help="生成点后立刻 decode + render")
    p.add_argument("--import-scores", action="store_true", help="导入评分表")
    p.add_argument("--score_csv", type=str, default="/root/autodl-tmp/MorphAny3D/output/scores_template.csv", help="评分表路径；默认使用 <out>/scores_template.csv")
    p.add_argument("--refine", action="store_true", help="根据当前评分继续细分")
    p.add_argument("--map-only", action="store_true", help="只生成地图，不做生成/细分")
    p.add_argument("--show-map-after-run", action="store_true", help="每次运行结束后都尝试生成地图")
    p.add_argument("--reset-state", action="store_true", help="清空当前 out 目录里的搜索状态重新开始")

    return p


def main() -> None:
    args = build_parser().parse_args()
    explorer = LatentGridExplorer(args)
    explorer.run_all()


if __name__ == "__main__":
    main()
