from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import yaml
from fvcore.common.config import CfgNode

from lib.dynamic_radio_dataset import DynamicRadioSingleFrameDiffusionDataset
from scripts.probe_dynamic_radio_ldm_btt_batch import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/dynamic_radio_ldm_btt.yaml")
    parser.add_argument("--ckpt", default="results/dynamic_radio_ldm_btt_fulltrain/model-39.pt")
    parser.add_argument("--out-dir", default="results/dynamic_radio_ldm_btt_fulltrain/video_model_39_val_record0")
    parser.add_argument("--split", default="val")
    parser.add_argument("--record-index", type=int, default=0)
    parser.add_argument("--num-frames", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260608)
    return parser.parse_args()


def to_uint8(image: torch.Tensor) -> np.ndarray:
    array = image.detach().float().cpu().numpy()
    array = np.squeeze(array)
    return np.asarray(np.clip(array * 255.0, 0, 255), dtype=np.uint8)


def panel_gray(image: np.ndarray, title: str, scale: int) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    rgb = cv2.resize(rgb, (rgb.shape[1] * scale, rgb.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    cv2.putText(rgb, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(rgb, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    return rgb


def panel_error(error: np.ndarray, title: str, scale: int) -> np.ndarray:
    heat = np.asarray(np.clip(error * 255.0, 0, 255), dtype=np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    heat = cv2.resize(heat, (heat.shape[1] * scale, heat.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    cv2.putText(heat, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(heat, title, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    return heat


def make_frame(traffic: torch.Tensor, gt: torch.Tensor, pred: torch.Tensor, frame_id: int, scale: int) -> np.ndarray:
    traffic_u8 = to_uint8(traffic)
    gt_u8 = to_uint8(gt)
    pred_u8 = to_uint8(pred)
    error = np.abs(pred.detach().float().cpu().numpy().squeeze() - gt.detach().float().cpu().numpy().squeeze())

    panels = [
        panel_gray(traffic_u8, f"traffic {frame_id:03d}", scale),
        panel_gray(gt_u8, "ground truth", scale),
        panel_gray(pred_u8, "prediction", scale),
        panel_error(error, "abs error", scale),
    ]
    return np.concatenate(panels, axis=1)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = CfgNode(yaml.load(open(args.cfg), Loader=yaml.FullLoader))
    out_dir = Path(args.out_dir)
    frame_dir = out_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)

    dataset = DynamicRadioSingleFrameDiffusionDataset(
        root=cfg.data.data_root,
        split=args.split,
        split_file=cfg.data.get("split_file", "split.json"),
        source=cfg.data.get("source", "png"),
        frame_stride=cfg.data.get("frame_stride", 1),
        cache_size=cfg.data.get("cache_size", 8),
        tx_heatmap_sigma_px=cfg.data.get("tx_heatmap_sigma_px", 1.5),
    )
    if args.record_index < 0 or args.record_index >= len(dataset.records):
        raise IndexError(f"record-index must be in [0, {len(dataset.records)})")

    record = dataset.records[args.record_index]
    frame_count = len(dataset.frame_index) // max(1, len(dataset.records))
    num_frames = min(args.num_frames, frame_count)
    base_index = args.record_index * frame_count
    indices = [base_index + i for i in range(num_frames)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    video_path = out_dir / "episode_prediction.mp4"
    meta = {
        "checkpoint": str(Path(args.ckpt).resolve()),
        "config": str(Path(args.cfg).resolve()),
        "split": args.split,
        "record_index": args.record_index,
        "scene_id": record.scene_id,
        "episode_id": record.episode_id,
        "tx_id": record.tx_id,
        "num_frames": num_frames,
        "fps": args.fps,
        "batch_size": args.batch_size,
        "video": str(video_path.resolve()),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    expected_frames = [frame_dir / f"frame_{frame_id:03d}.png" for frame_id in range(num_frames)]
    if not all(path.exists() for path in expected_frames):
        with torch.no_grad():
            for start in range(0, len(indices), args.batch_size):
                batch_indices = indices[start : start + args.batch_size]
                samples = [dataset[index] for index in batch_indices]
                cond = torch.stack([item["cond"] for item in samples]).to(device)
                gt = torch.stack([item["image"] for item in samples]).to(device)
                gt_vis = torch.clamp((gt + 1.0) / 2.0, 0.0, 1.0)
                traffic_vis = torch.clamp((cond[:, 2:3] + 1.0) / 2.0, 0.0, 1.0)
                pred = torch.clamp(model.sample(batch_size=cond.shape[0], cond=cond), 0.0, 1.0)

                for offset in range(cond.shape[0]):
                    frame_id = start + offset
                    canvas = make_frame(traffic_vis[offset], gt_vis[offset], pred[offset], frame_id, args.scale)
                    imageio.imwrite(frame_dir / f"frame_{frame_id:03d}.png", canvas)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg executable not found in PATH")
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-framerate",
            str(args.fps),
            "-i",
            str(frame_dir / "frame_%03d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            str(video_path),
        ],
        check=True,
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
