# DynamicRadioMap 训练复刻交接文档

本文档给 PINN 和 RadioDiff-K 两个代码窗口使用，目标是在它们各自仓库中复刻当前 RadioDiff 的 DynamicRadioMap 训练/评估流程，用于公平比较不同模型在同一动态数据集上的性能。

## 1. 当前任务定义

原始 RadioMapSeer 是静态地图数据；当前实验使用的是 CARLA 生成的动态 RadioMapSeer-style 数据包：

```text
/data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack
```

核心任务：

```text
输入条件 cond = [building_mask, tx_heatmap, traffic_frame]
输出目标 image = dynamic RSS/radio map frame
```

当前 RadioDiff 训练是两阶段：

```text
1. VAE/AutoencoderKL: 单帧 dynamic RSS 自编码，输入 image，输出重建 image。
2. Conditional latent diffusion: 输入 cond，生成 image 的 VAE latent，再由 VAE decoder 解码。
```

PINN 和 RadioDiff-K 如果原本也是 RadioMapSeer loader，可以优先复用其模型/损失，只替换 dataset 和 config，使输入/输出与这里一致。

## 2. 数据包结构

数据根目录下的关键文件：

```text
dataset_meta.json
index.json
index.jsonl
split.json
splits.json
split_mixed_50_50.json
scenes/
```

单个样本大致位于：

```text
scenes/{scene_id}/episodes/{episode_id}/tx/{tx_id}/sample_meta.json
```

对应资源：

```text
scenes/{scene_id}/building/building_mask.npz
scenes/{scene_id}/static/{tx_id}/static_rss.npz
scenes/{scene_id}/episodes/{episode_id}/traffic/traffic_grid_uint8.npz
scenes/{scene_id}/episodes/{episode_id}/frame_indices.npy
scenes/{scene_id}/episodes/{episode_id}/tx/{tx_id}/rss_maps.npz
scenes/{scene_id}/episodes/{episode_id}/tx/{tx_id}/png/frame_000000.png
...
```

当前实际 split 规模：

```text
train: 900000 single frames, 9000 tx samples
val:    75000 single frames,  750 tx samples
test:   75000 single frames,  750 tx samples
```

`split.json` 用于 LDM/full train；`split_mixed_50_50.json` 曾用于 VAE mixed 微调。

## 3. 推荐直接复用的文件

从当前 RadioDiff 仓库复制到目标仓库：

```text
lib/dynamic_radio_dataset.py
configs/first_dynamic_radio.yaml
configs/first_dynamic_radio_mixed_50_50.yaml
configs/dynamic_radio_ldm_btt.yaml
scripts/eval_dynamic_radio_ldm_val10.py
scripts/eval_dynamic_radio_ldm_checkpoints.py
```

最关键的是：

```text
lib/dynamic_radio_dataset.py
```

它提供两个 Dataset：

```python
DynamicRadioSingleFrameVaeDataset
DynamicRadioSingleFrameDiffusionDataset
```

## 4. Dataset 输出约定

### 4.1 VAE dataset

类：

```python
DynamicRadioSingleFrameVaeDataset
```

返回：

```python
{
    "image": Tensor[1, 128, 128],  # dynamic RSS frame, normalized to [-1, 1]
    "img_name": str,
}
```

用途：训练第一阶段 VAE。

### 4.2 Conditional model dataset

类：

```python
DynamicRadioSingleFrameDiffusionDataset
```

返回：

```python
{
    "image": Tensor[1, 128, 128],  # target RSS, normalized to [-1, 1]
    "cond":  Tensor[3, 128, 128],  # condition, normalized to [-1, 1]
    "img_name": str,
}
```

`cond` 三通道定义：

```text
cond[0] = building mask, 0/1 then mapped to [-1, 1]
cond[1] = tx gaussian heatmap, sigma_px=1.5, then mapped to [-1, 1]
cond[2] = traffic frame, uint8/255 then mapped to [-1, 1]
```

target `image`：

```text
dynamic RSS png/npz uint8 -> /255 -> [0, 1] -> *2-1 -> [-1, 1]
```

如果 PINN 或 RadioDiff-K 原代码期望 `[0, 1]`，需要在模型入口或 loss 前统一转换，不要混用。

## 5. 训练配置参考

### 5.1 VAE 初训

参考：

```text
configs/first_dynamic_radio.yaml
```

关键参数：

```yaml
model:
  embed_dim: 3
  ddconfig:
    resolution: [128, 128]
    in_channels: 1
    out_ch: 1
    z_channels: 3
    ch: 128
    ch_mult: [1, 2, 4]
  lossconfig:
    disc_start: 50001
    kl_weight: 0.000001
    disc_weight: 0.5

data:
  name: dynamic_radio_single_frame_vae
  data_root: /data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack
  split_file: split.json
  split: train
  source: png
  batch_size: 20

trainer:
  gradient_accumulate_every: 2
  lr: 5e-6
  train_num_steps: 150000
  save_and_sample_every: 5000
  auto_resume: true
```

当前 RadioDiff 已训练可用 VAE：

```text
results/dynamic_radio_vae_mixed_50_50/model-20-raw-model-only.pt
```

如果另外两个仓库不需要重新训练 VAE，可以直接使用这个 VAE 权重作为 latent autoencoder。若模型不是 latent diffusion，则可跳过 VAE，直接使用 `DynamicRadioSingleFrameDiffusionDataset` 做 image-to-image 训练。

### 5.2 LDM/条件生成训练

参考：

```text
configs/dynamic_radio_ldm_btt.yaml
```

关键参数：

```yaml
model:
  image_size: [128, 128]
  sampling_timesteps: 50
  objective: pred_KC
  first_stage:
    ckpt_path: ./results/dynamic_radio_vae_mixed_50_50/model-20-raw-model-only.pt
    ddconfig:
      resolution: [128, 128]
      z_channels: 3
  unet:
    channels: 3
    cond_in_dim: 3
    cond_feature_size: [32, 32]
    input_size: [32, 32]

data:
  name: dynamic_radio_btt_cond
  split_file: split.json
  split: train
  batch_size: 56

trainer:
  gradient_accumulate_every: 4
  lr: 5e-5
  train_num_steps: 50000
  save_and_sample_every: 1000
```

注意 latent 尺寸：

```text
image: 128x128
VAE downsample factor: 4
latent/input_size: 32x32
```

如果 PINN/RadioDiff-K 的网络内部写死了 `256`、`320` 或 latent `64/80`，需要改为 `128` 和 `32`。

## 6. 目标仓库最小改造步骤

### 6.1 添加 dataset

把 `lib/dynamic_radio_dataset.py` 复制过去，并在训练入口里加分支。

VAE 训练入口：

```python
from lib.dynamic_radio_dataset import DynamicRadioSingleFrameVaeDataset

dataset = DynamicRadioSingleFrameVaeDataset(
    root=data_cfg["data_root"],
    split=data_cfg.get("split", "train"),
    split_file=data_cfg.get("split_file", "split.json"),
    source=data_cfg.get("source", "png"),
    frame_stride=data_cfg.get("frame_stride", 1),
    cache_size=data_cfg.get("cache_size", 8),
)
```

条件模型训练入口：

```python
from lib.dynamic_radio_dataset import DynamicRadioSingleFrameDiffusionDataset

dataset = DynamicRadioSingleFrameDiffusionDataset(
    root=data_cfg["data_root"],
    split=data_cfg.get("split", "train"),
    split_file=data_cfg.get("split_file", "split.json"),
    source=data_cfg.get("source", "png"),
    frame_stride=data_cfg.get("frame_stride", 1),
    cache_size=data_cfg.get("cache_size", 8),
    tx_heatmap_sigma_px=data_cfg.get("tx_heatmap_sigma_px", 1.5),
)
```

### 6.2 对齐 batch dict

当前 batch keys：

```text
image: target RSS
cond: condition image
img_name: path-like identifier
```

如果目标仓库原本用：

```text
input / target
img / mask
radio / building / tx
```

请在训练 loop 入口做一次映射，建议不要改 dataset 输出，方便跨仓库一致评估。

### 6.3 对齐归一化

训练时：

```python
image in [-1, 1]
cond in [-1, 1]
```

评估/可视化时：

```python
gt = (batch["image"] + 1.0) / 2.0
pred = clamp(pred, 0, 1)
```

如果模型直接输出 `[-1, 1]`，评估前先转到 `[0, 1]`。

## 7. 评估方式

快速可视化：

```bash
python scripts/eval_dynamic_radio_ldm_val10.py \
  --cfg configs/dynamic_radio_ldm_btt.yaml \
  --ckpt results/dynamic_radio_ldm_btt_fulltrain/model-50.pt \
  --out-dir results/eval_model_50_val10 \
  --num-samples 10 \
  --split val \
  --selection diverse
```

checkpoint sweep：

```bash
python scripts/eval_dynamic_radio_ldm_checkpoints.py \
  --cfg configs/dynamic_radio_ldm_btt.yaml \
  --ckpt-dir results/dynamic_radio_ldm_btt_fulltrain \
  --out-dir results/dynamic_radio_ldm_btt_fulltrain/val_ckpt_eval_36_50_5000 \
  --split val \
  --selection diverse \
  --max-samples 5000 \
  --batch-size 64 \
  --start 36 \
  --end 50
```

输出：

```text
summary.csv
best.json
model-{id}.json
```

指标：

```text
MAE
MSE
RMSE
NMSE
global_NMSE
PSNR
```

目前 RadioDiff 这边正在做：

```text
model-36 到 model-50
val diverse 5000 frames
8 GPU parallel
batch_size 64
```

## 8. 已知结果参考

`model-31.pt` 在 val diverse 10 frames 上：

```text
MAE  = 0.0680
MSE  = 0.0181
NMSE = 0.0671
PSNR = 17.67 dB
```

这是非常小的抽样，只能作为 sanity check，不适合作最终模型选择。

最新 15 个 checkpoint 的 5000-frame sweep 仍在跑。结果目录：

```text
results/dynamic_radio_ldm_btt_fulltrain/val_ckpt_eval_36_50_5000
```

## 9. 常见踩坑

### 9.1 不要使用旧 RadioMapSeer 的 256/320 尺寸假设

当前动态数据是：

```text
image/cond: 128x128
latent: 32x32
```

### 9.2 cond 不是旧 RadioMapSeer 的 building/tx/building

当前 cond 是：

```text
building + tx_heatmap + traffic
```

第三通道不是重复 building，也不是 cars png，而是动态 traffic frame。

### 9.3 split 单位是 tx sample，训练单位是 frame

`split.json` 里保存的是 tx sample meta path；dataset 会展开为每个 tx sample 的 100 帧。

例如：

```text
750 val tx samples -> 75000 val frames
```

### 9.4 checkpoint 选择不要只看训练 loss

Diffusion 采样指标有随机性，建议：

```text
1. 先用 val 5000 frames 筛 model-36~50。
2. 再对 top 3/top 5 用 val 10000 frames 复核。
3. 最后用 test split 报最终指标。
```

### 9.5 数据读取源

推荐：

```yaml
source: png
```

`npz_uint8` 也支持，但不同仓库中读取大 npz 的缓存策略可能影响内存和速度。

## 10. 给 PINN/RadioDiff-K 的适配建议

### PINN

如果 PINN 需要额外物理约束输入，建议先跑 baseline：

```text
input = [building, tx_heatmap, traffic]
target = dynamic RSS
```

确认能训通后，再添加物理项。不要一开始同时改 dataset、loss、模型结构，否则很难定位问题。

### RadioDiff-K

如果原代码使用 K2/Helmholtz 相关输入，建议把旧 K2 输入作为额外通道实验：

```text
baseline cond: 3 channels = building, tx, traffic
K variant:     4/5 channels = building, tx, traffic, K2/physics prior
```

但第一版公平比较建议保持 3 通道，与当前 RadioDiff 结果一致。

## 11. 新服务器数据获取

代码：

```bash
git clone git@github.com:ambitious-rice/RadioDiff.git
cd RadioDiff
```

数据已上传到 GitHub Release：

```text
https://github.com/ambitious-rice/RadioDiff/releases/tag/dynamic-radiomap-pack-v1
```

下载并解包：

```bash
gh release download dynamic-radiomap-pack-v1 \
  --repo ambitious-rice/RadioDiff \
  --dir release_assets \
  --pattern 'MultiScene20_RF300M8Runs_RadioMapSeerPack.tar.gz*' \
  --skip-existing

./scripts/unpack_radiomapseer_release.sh \
  release_assets \
  /data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap
```

完整文件数：

```text
78 part files + 1 sha256 = 79 files
```

解包后路径：

```text
/data/fzj/CARLA_0.9.15/datasets/DynamicRadioMap/MultiScene20_RF300M8Runs_RadioMapSeerPack
```

