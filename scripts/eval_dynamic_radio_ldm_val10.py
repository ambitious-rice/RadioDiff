from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torchvision.utils as vutils
import yaml
from fvcore.common.config import CfgNode
from torch.utils.data import DataLoader, Subset

from lib.dynamic_radio_dataset import DynamicRadioSingleFrameDiffusionDataset
from scripts.probe_dynamic_radio_ldm_btt_batch import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/dynamic_radio_ldm_btt.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--split", default="val")
    parser.add_argument("--selection", choices=["first", "diverse", "random"], default="diverse")
    parser.add_argument("--seed", type=int, default=20260605)
    return parser.parse_args()


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(torch.clamp(mse, min=1e-12))


def select_indices(
    dataset: DynamicRadioSingleFrameDiffusionDataset,
    num_samples: int,
    mode: str,
    seed: int,
) -> list[int]:
    if mode == "first":
        return list(range(min(num_samples, len(dataset))))
    if mode == "random":
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        count = min(num_samples, len(dataset))
        return torch.randperm(len(dataset), generator=generator)[:count].tolist()

    frame_count = len(dataset.frame_index) // max(1, len(dataset.records))
    if frame_count <= 0:
        return list(range(min(num_samples, len(dataset))))

    selected: list[int] = []
    record_ids = torch.linspace(0, len(dataset.records) - 1, steps=min(num_samples, len(dataset.records))).round().long()
    frame_ids = torch.linspace(0, frame_count - 1, steps=len(record_ids)).round().long()
    seen: set[int] = set()
    for record_id, frame_id in zip(record_ids.tolist(), frame_ids.tolist()):
        index = int(record_id) * frame_count + int(frame_id)
        if index not in seen and index < len(dataset):
            selected.append(index)
            seen.add(index)

    index = 0
    while len(selected) < min(num_samples, len(dataset)) and index < len(dataset):
        if index not in seen:
            selected.append(index)
            seen.add(index)
        index += max(1, frame_count)
    return selected[:num_samples]


def main() -> None:
    args = parse_args()
    cfg = CfgNode(yaml.load(open(args.cfg), Loader=yaml.FullLoader))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = DynamicRadioSingleFrameDiffusionDataset(
        root=cfg.data.data_root,
        split=args.split,
        split_file=cfg.data.get("split_file", "split.json"),
        source=cfg.data.get("source", "png"),
        frame_stride=cfg.data.get("frame_stride", 1),
        cache_size=cfg.data.get("cache_size", 8),
        tx_heatmap_sigma_px=cfg.data.get("tx_heatmap_sigma_px", 1.5),
    )
    selected_indices = select_indices(dataset, args.num_samples, args.selection, args.seed)
    subset = Subset(dataset, selected_indices)
    loader = DataLoader(subset, batch_size=len(subset), shuffle=False, num_workers=2, pin_memory=True)
    batch = next(iter(loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    cond = batch["cond"].to(device, non_blocking=True)
    gt = (batch["image"].to(device, non_blocking=True) + 1.0) / 2.0

    with torch.no_grad():
        pred = model.sample(batch_size=cond.shape[0], cond=cond)

    pred = torch.clamp(pred, 0.0, 1.0)
    gt = torch.clamp(gt, 0.0, 1.0)
    cond_vis = (cond.detach().cpu() + 1.0) / 2.0

    vutils.save_image(gt.cpu(), out_dir / "val10_gt.png", nrow=5)
    vutils.save_image(pred.cpu(), out_dir / "val10_pred.png", nrow=5)
    vutils.save_image(cond_vis[:, 0:1], out_dir / "val10_cond_building.png", nrow=5)
    vutils.save_image(cond_vis[:, 1:2], out_dir / "val10_cond_tx.png", nrow=5)
    vutils.save_image(cond_vis[:, 2:3], out_dir / "val10_cond_traffic.png", nrow=5)
    vutils.save_image(torch.cat([gt.cpu(), pred.cpu()], dim=0), out_dir / "val10_gt_then_pred.png", nrow=5)

    err = pred - gt
    per_mse = err.square().mean(dim=(1, 2, 3))
    per_mae = err.abs().mean(dim=(1, 2, 3))
    per_nmse = err.square().sum(dim=(1, 2, 3)) / torch.clamp(gt.square().sum(dim=(1, 2, 3)), min=1e-12)
    per_psnr = psnr_from_mse(per_mse)

    metrics = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "config": str(Path(args.cfg).resolve()),
        "split_file": cfg.data.get("split_file", "split.json"),
        "split": args.split,
        "num_samples": int(gt.shape[0]),
        "selection": args.selection,
        "seed": int(args.seed),
        "selected_indices": selected_indices,
        "image_names": list(batch["img_name"]),
        "mae": float(per_mae.mean().cpu()),
        "mse": float(per_mse.mean().cpu()),
        "nmse": float(per_nmse.mean().cpu()),
        "psnr": float(per_psnr.mean().cpu()),
        "per_sample": [
            {
                "img_name": str(batch["img_name"][idx]),
                "mae": float(per_mae[idx].cpu()),
                "mse": float(per_mse[idx].cpu()),
                "nmse": float(per_nmse[idx].cpu()),
                "psnr": float(per_psnr[idx].cpu()),
            }
            for idx in range(gt.shape[0])
        ],
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
