from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torchvision as tv
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL


@dataclass(frozen=True)
class EvalItem:
    scene_id: str
    sample_meta: str
    frame_idx: int
    image_path: Path


class DynamicRadioIndexFrameDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        scenes: list[str],
        samples_per_scene: int,
        frame_count: int = 100,
        seed: int = 20260604,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.scenes = set(scenes)
        self.samples_per_scene = int(samples_per_scene)
        self.frame_count = int(frame_count)
        self.seed = int(seed)
        self.items = self._build_items()

    def _build_items(self) -> list[EvalItem]:
        index = json.loads((self.root / "index.json").read_text())
        by_scene: dict[str, list[dict[str, Any]]] = {scene: [] for scene in self.scenes}
        for sample in index["samples"]:
            scene_id = sample["scene_id"]
            if scene_id in by_scene:
                by_scene[scene_id].append(sample)

        items: list[EvalItem] = []
        for scene_id in sorted(by_scene):
            samples = by_scene[scene_id]
            if not samples:
                raise ValueError(f"No samples found for scene {scene_id!r}")
            rng = random.Random(self.seed + sum(ord(ch) for ch in scene_id))
            candidates = [
                (sample, frame_idx)
                for sample in samples
                for frame_idx in range(self.frame_count)
            ]
            chosen = rng.sample(candidates, min(self.samples_per_scene, len(candidates)))
            for sample, frame_idx in chosen:
                sample_meta = sample["sample_meta"]
                tx_dir = Path(sample_meta).parent
                items.append(
                    EvalItem(
                        scene_id=scene_id,
                        sample_meta=sample_meta,
                        frame_idx=frame_idx,
                        image_path=self.root / tx_dir / "png" / f"frame_{frame_idx:06d}.png",
                    )
                )
        return items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self.items[int(idx)]
        with Image.open(item.image_path) as image:
            array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).unsqueeze(0).mul(2.0).sub(1.0).contiguous()
        return {
            "image": tensor,
            "scene_id": item.scene_id,
            "sample_meta": item.sample_meta,
            "frame_idx": item.frame_idx,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/first_dynamic_radio.yaml")
    parser.add_argument("--results-folder", default="results/dynamic_radio_vae")
    parser.add_argument("--first-model", type=int, default=10)
    parser.add_argument("--last-model", type=int, default=17)
    parser.add_argument("--samples-per-scene", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260604)
    parser.add_argument("--out-dir", default="results/dynamic_radio_vae/new_scene_eval")
    return parser.parse_args()


def discover_new_scenes(root: Path) -> list[str]:
    index = json.loads((root / "index.json").read_text())
    split = json.loads((root / "split.json").read_text())
    indexed_scenes = {sample["scene_id"] for sample in index["samples"]}
    old_train_scenes = set(split["scene_ids"]["train"])
    return sorted(indexed_scenes - old_train_scenes)


def init_model(cfg: dict[str, Any], checkpoint: Path, device: torch.device) -> AutoencoderKL:
    model = AutoencoderKL(
        ddconfig=cfg["model"]["ddconfig"],
        lossconfig=cfg["model"]["lossconfig"],
        embed_dim=cfg["model"]["embed_dim"],
        ckpt_path=None,
    )
    data = torch.load(str(checkpoint), map_location="cpu")
    model.load_state_dict(data["model"])
    model.to(device)
    model.eval()
    return model


def update_metrics(acc: dict[str, float], pred: torch.Tensor, target: torch.Tensor) -> None:
    pred01 = torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)
    target01 = torch.clamp((target + 1.0) / 2.0, 0.0, 1.0)
    diff = pred01 - target01
    abs_sum = diff.abs().sum().item()
    sq_sum = diff.square().sum().item()
    gt_sq_sum = target01.square().sum().item()
    pixels = float(diff.numel())
    acc["abs_sum"] += abs_sum
    acc["sq_sum"] += sq_sum
    acc["gt_sq_sum"] += gt_sq_sum
    acc["pixels"] += pixels
    acc["count"] += float(diff.shape[0])


def finalize_metrics(acc: dict[str, float]) -> dict[str, float]:
    mae = acc["abs_sum"] / acc["pixels"]
    mse = acc["sq_sum"] / acc["pixels"]
    rmse = math.sqrt(mse)
    psnr = 10.0 * math.log10(1.0 / max(mse, 1e-12))
    nmse = acc["sq_sum"] / max(acc["gt_sq_sum"], 1e-12)
    return {
        "count": int(acc["count"]),
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "psnr": psnr,
        "nmse": nmse,
    }


def new_acc() -> dict[str, float]:
    return {"abs_sum": 0.0, "sq_sum": 0.0, "gt_sq_sum": 0.0, "pixels": 0.0, "count": 0.0}


def main() -> None:
    args = parse_args()
    cfg = yaml.load(open(args.cfg), Loader=yaml.FullLoader)
    root = Path(cfg["data"]["data_root"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scenes = discover_new_scenes(root)
    dataset = DynamicRadioIndexFrameDataset(
        root=root,
        scenes=scenes,
        samples_per_scene=args.samples_per_scene,
        seed=args.seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    meta = {
        "cfg": args.cfg,
        "data_root": str(root),
        "new_scenes": scenes,
        "samples_per_scene": args.samples_per_scene,
        "total_samples": len(dataset),
        "seed": args.seed,
        "models": list(range(args.first_model, args.last_model + 1)),
    }
    (out_dir / "eval_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    results: list[dict[str, Any]] = []
    for model_id in range(args.first_model, args.last_model + 1):
        checkpoint = Path(args.results_folder) / f"model-{model_id}.pt"
        print(f"Evaluating {checkpoint} on {len(dataset)} frames")
        model = init_model(cfg, checkpoint, device)
        overall = new_acc()
        per_scene = {scene: new_acc() for scene in scenes}
        saved_preview = False

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                image = batch["image"].to(device, non_blocking=True)
                recon = model.validate_img(image)
                update_metrics(overall, recon, image)
                for scene_id in sorted(set(batch["scene_id"])):
                    mask = [value == scene_id for value in batch["scene_id"]]
                    idx = torch.tensor(mask, dtype=torch.bool, device=device)
                    update_metrics(per_scene[scene_id], recon[idx], image[idx])
                if not saved_preview:
                    preview = torch.cat((image[:2], recon[:2]), dim=0)
                    preview = torch.clamp((preview + 1.0) / 2.0, 0.0, 1.0)
                    tv.utils.save_image(preview, str(out_dir / f"sample-model-{model_id}.png"), nrow=2)
                    saved_preview = True

        row = {"model": model_id, **finalize_metrics(overall)}
        results.append(row)
        per_scene_rows = [
            {"model": model_id, "scene_id": scene, **finalize_metrics(acc)}
            for scene, acc in per_scene.items()
        ]
        (out_dir / f"per_scene_model_{model_id}.json").write_text(
            json.dumps(per_scene_rows, indent=2), encoding="utf-8"
        )
        print(json.dumps(row, indent=2))
        del model
        torch.cuda.empty_cache()

    (out_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
