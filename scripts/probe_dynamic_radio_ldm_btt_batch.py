from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import yaml
from fvcore.common.config import CfgNode
from torch.utils.data import DataLoader

from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
from lib.dynamic_radio_dataset import DynamicRadioSingleFrameDiffusionDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", default="configs/dynamic_radio_ldm_btt.yaml")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[4, 8, 12, 16, 20, 24, 28, 32])
    parser.add_argument("--grad-accum", type=int, default=1)
    return parser.parse_args()


def build_model(cfg: CfgNode):
    first_stage_cfg = cfg.model.first_stage
    first_stage_model = AutoencoderKL(
        ddconfig=first_stage_cfg.ddconfig,
        lossconfig=first_stage_cfg.lossconfig,
        embed_dim=first_stage_cfg.embed_dim,
        ckpt_path=first_stage_cfg.ckpt_path,
    )

    from denoising_diffusion_pytorch.mask_cond_unet import Unet
    unet_cfg = cfg.model.unet
    unet = Unet(
        dim=unet_cfg.dim,
        channels=unet_cfg.channels,
        dim_mults=unet_cfg.dim_mults,
        learned_variance=unet_cfg.get("learned_variance", False),
        out_mul=unet_cfg.out_mul,
        cond_in_dim=unet_cfg.cond_in_dim,
        cond_dim=unet_cfg.cond_dim,
        cond_dim_mults=unet_cfg.cond_dim_mults,
        window_sizes1=unet_cfg.window_sizes1,
        window_sizes2=unet_cfg.window_sizes2,
        fourier_scale=unet_cfg.fourier_scale,
        carsDPM=unet_cfg.get("DPMCARK", False),
        cfg=unet_cfg,
    )

    from denoising_diffusion_pytorch.ddm_const_sde import LatentDiffusion
    return LatentDiffusion(
        model=unet,
        auto_encoder=first_stage_model,
        train_sample=cfg.model.train_sample,
        image_size=cfg.model.image_size,
        timesteps=cfg.model.timesteps,
        sampling_timesteps=cfg.model.sampling_timesteps,
        loss_type=cfg.model.loss_type,
        objective=cfg.model.objective,
        scale_factor=cfg.model.scale_factor,
        scale_by_std=cfg.model.scale_by_std,
        scale_by_softsign=cfg.model.scale_by_softsign,
        default_scale=cfg.model.get("default_scale", False),
        input_keys=cfg.model.input_keys,
        ckpt_path=cfg.model.ckpt_path,
        ignore_keys=cfg.model.ignore_keys,
        only_model=cfg.model.only_model,
        start_dist=cfg.model.start_dist,
        perceptual_weight=cfg.model.perceptual_weight,
        use_l1=cfg.model.get("use_l1", True),
        cfg=cfg.model,
    )


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
            model.train()
            opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.trainer.lr)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
            opt.zero_grad(set_to_none=True)
            loader_iter = iter(loader)
            total_loss = 0.0
            for _ in range(args.grad_accum):
                batch = next(loader_iter)
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(device, non_blocking=True)
                loss, loss_dict = model.training_step(batch)
                (loss / args.grad_accum).backward()
                total_loss += float(loss.detach().cpu())
            opt.step()
            torch.cuda.synchronize(device)
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            reserved = torch.cuda.max_memory_reserved(device) / 1024**3
            print(
                f"OK batch_size={batch_size} "
                f"grad_accum={args.grad_accum} "
                f"loss={total_loss / args.grad_accum:.6f} "
                f"peak_alloc_gb={peak:.2f} peak_reserved_gb={reserved:.2f}",
                flush=True,
            )
            del model, opt, loader, batch, loss, loss_dict
        except RuntimeError as err:
            if "out of memory" not in str(err).lower():
                raise
            print(f"OOM batch_size={batch_size}", flush=True)
            break
        finally:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
