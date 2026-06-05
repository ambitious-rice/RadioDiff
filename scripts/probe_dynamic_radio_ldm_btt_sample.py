from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from fvcore.common.config import CfgNode
from torch.utils.data import DataLoader

from probe_dynamic_radio_ldm_btt_batch import build_model
from lib.dynamic_radio_dataset import DynamicRadioSingleFrameDiffusionDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/dynamic_radio_ldm_btt.yaml")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[16, 32, 48, 64, 80])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CfgNode(yaml.load(open(args.cfg), Loader=yaml.FullLoader))
    data_cfg = cfg.data
    dataset = DynamicRadioSingleFrameDiffusionDataset(
        root=data_cfg.data_root,
        split=data_cfg.get("split", "train"),
        split_file=data_cfg.get("split_file", "split.json"),
        source=data_cfg.get("source", "png"),
        frame_stride=data_cfg.get("frame_stride", 1),
        cache_size=data_cfg.get("cache_size", 8),
        tx_heatmap_sigma_px=data_cfg.get("tx_heatmap_sigma_px", 1.5),
    )

    device = torch.device("cuda")
    for batch_size in args.batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        try:
            model = build_model(cfg).to(device)
            model.eval()
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            batch = next(iter(loader))
            cond = batch["cond"].to(device, non_blocking=True)
            with torch.no_grad():
                sample = model.sample(batch_size=cond.shape[0], cond=cond)
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            reserved = torch.cuda.max_memory_reserved(device) / 1024**3
            print(
                f"OK sample_batch_size={batch_size} "
                f"sample_shape={tuple(sample.shape)} "
                f"peak_alloc_gb={peak:.2f} peak_reserved_gb={reserved:.2f}",
                flush=True,
            )
            del model, loader, batch, cond, sample
        except RuntimeError as err:
            if "out of memory" not in str(err).lower():
                raise
            print(f"OOM sample_batch_size={batch_size}", flush=True)
            break
        finally:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
