from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from fvcore.common.config import CfgNode
from torch.utils.data import DataLoader, Subset

from lib.dynamic_radio_dataset import DynamicRadioSingleFrameDiffusionDataset
from scripts.eval_dynamic_radio_ldm_val10 import select_indices
from scripts.probe_dynamic_radio_ldm_btt_batch import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/dynamic_radio_ldm_btt.yaml")
    parser.add_argument("--ckpt-dir", default="results/dynamic_radio_ldm_btt_fulltrain")
    parser.add_argument("--out-dir", default="results/dynamic_radio_ldm_btt_fulltrain/val_checkpoint_eval")
    parser.add_argument("--split", default="val")
    parser.add_argument("--selection", choices=["full", "first", "diverse", "random"], default="full")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260607)
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=10**9)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def checkpoint_id(path: Path) -> int:
    match = re.fullmatch(r"model-(\d+)\.pt", path.name)
    if not match:
        raise ValueError(f"Unexpected checkpoint name: {path}")
    return int(match.group(1))


def list_checkpoints(ckpt_dir: Path, start: int, end: int, num_shards: int, shard_index: int) -> list[Path]:
    ckpts = sorted(ckpt_dir.glob("model-*.pt"), key=checkpoint_id)
    selected = [p for p in ckpts if start <= checkpoint_id(p) <= end]
    return [p for idx, p in enumerate(selected) if idx % num_shards == shard_index]


def make_loader(args: argparse.Namespace, cfg: CfgNode) -> tuple[DataLoader, int, list[int] | None]:
    dataset = DynamicRadioSingleFrameDiffusionDataset(
        root=cfg.data.data_root,
        split=args.split,
        split_file=cfg.data.get("split_file", "split.json"),
        source=cfg.data.get("source", "png"),
        frame_stride=cfg.data.get("frame_stride", 1),
        cache_size=cfg.data.get("cache_size", 8),
        tx_heatmap_sigma_px=cfg.data.get("tx_heatmap_sigma_px", 1.5),
    )

    indices: list[int] | None = None
    if args.selection != "full" or args.max_samples > 0:
        count = args.max_samples if args.max_samples > 0 else len(dataset)
        mode = "diverse" if args.selection == "full" else args.selection
        indices = select_indices(dataset, count, mode, args.seed)
        dataset = Subset(dataset, indices)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return loader, len(dataset), indices


def psnr_from_mse(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(torch.clamp(mse, min=1e-12))


def evaluate_checkpoint(ckpt: Path, cfg: CfgNode, loader: DataLoader, device: torch.device) -> dict[str, object]:
    model = build_model(cfg).to(device)
    checkpoint = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    sample_count = 0
    pixel_count = 0
    abs_sum = 0.0
    sq_sum = 0.0
    gt_sq_sum = 0.0
    nmse_sum = 0.0
    psnr_sum = 0.0

    with torch.no_grad():
        for batch in loader:
            cond = batch["cond"].to(device, non_blocking=True)
            gt = (batch["image"].to(device, non_blocking=True) + 1.0) / 2.0
            pred = model.sample(batch_size=cond.shape[0], cond=cond)
            pred = torch.clamp(pred, 0.0, 1.0)
            gt = torch.clamp(gt, 0.0, 1.0)

            err = pred - gt
            per_mse = err.square().mean(dim=(1, 2, 3))
            per_nmse = err.square().sum(dim=(1, 2, 3)) / torch.clamp(gt.square().sum(dim=(1, 2, 3)), min=1e-12)
            per_psnr = psnr_from_mse(per_mse)

            batch_samples = int(gt.shape[0])
            sample_count += batch_samples
            pixel_count += int(gt.numel())
            abs_sum += float(err.abs().sum().detach().cpu())
            sq_sum += float(err.square().sum().detach().cpu())
            gt_sq_sum += float(gt.square().sum().detach().cpu())
            nmse_sum += float(per_nmse.sum().detach().cpu())
            psnr_sum += float(per_psnr.sum().detach().cpu())

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mse = sq_sum / max(pixel_count, 1)
    return {
        "checkpoint": str(ckpt.resolve()),
        "model_id": checkpoint_id(ckpt),
        "samples": sample_count,
        "pixels": pixel_count,
        "mae": abs_sum / max(pixel_count, 1),
        "mse": mse,
        "rmse": mse**0.5,
        "nmse": nmse_sum / max(sample_count, 1),
        "global_nmse": sq_sum / max(gt_sq_sum, 1e-12),
        "psnr": psnr_sum / max(sample_count, 1),
    }


def write_summary(out_dir: Path) -> None:
    metrics = []
    for path in sorted(out_dir.glob("model-*.json"), key=lambda p: checkpoint_id(Path(p.stem + ".pt"))):
        metrics.append(json.loads(path.read_text(encoding="utf-8")))
    if not metrics:
        return

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    best_nmse = min(metrics, key=lambda row: row["nmse"])
    best_mse = min(metrics, key=lambda row: row["mse"])
    best_psnr = max(metrics, key=lambda row: row["psnr"])
    (out_dir / "best.json").write_text(
        json.dumps({"best_nmse": best_nmse, "best_mse": best_mse, "best_psnr": best_psnr}, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")

    cfg = CfgNode(yaml.load(open(args.cfg), Loader=yaml.FullLoader))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    loader, dataset_len, indices = make_loader(args, cfg)
    meta = {
        "config": str(Path(args.cfg).resolve()),
        "split": args.split,
        "selection": args.selection,
        "max_samples": args.max_samples,
        "dataset_len": dataset_len,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "start": args.start,
        "end": args.end,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "indices": indices,
    }
    (out_dir / f"meta-shard-{args.shard_index}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpts = list_checkpoints(Path(args.ckpt_dir), args.start, args.end, args.num_shards, args.shard_index)
    print(f"Evaluating {len(ckpts)} checkpoints on {device}; dataset_len={dataset_len}", flush=True)

    for ckpt in ckpts:
        metric_path = out_dir / f"model-{checkpoint_id(ckpt)}.json"
        if metric_path.exists() and not args.overwrite:
            print(f"Skip existing {metric_path}", flush=True)
            continue
        print(f"Evaluating {ckpt}", flush=True)
        metrics = evaluate_checkpoint(ckpt, cfg, loader, device)
        metric_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        write_summary(out_dir)
        print(json.dumps(metrics, indent=2), flush=True)

    write_summary(out_dir)


if __name__ == "__main__":
    main()
