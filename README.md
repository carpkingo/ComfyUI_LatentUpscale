# DragonBro Latent Upscale

两个 ComfyUI 自定义节点，解决 **latent（潜空间）放大时出现的"偏移 / 人物漂移 / 重影"** 问题，支持图片（4 维 latent）与视频（5 维 latent）。

- 分类：`DragonBro/Upscale`
- 类名（API class_type）：`LatentUpscaleByEnhanced` / `LatentHiresReencode`

---

## 一、节点一览

| 显示名（GUI） | 类名（class_type） | 作用 |
| --- | --- | --- |
| `DragonBro Latent Upscale By (Enhanced)` | `LatentUpscaleByEnhanced` | bislerp 增强版 latent 按系数放大，替代内置 `LatentUpscaleBy`，减少偏移与硬边 |
| `DragonBro Latent Hires Reencode` | `LatentHiresReencode` | decode → 像素放大（可接超分模型）→ encode 的 hires 重编码，适合像素级高清 |

---

## 二、安装

1. 把本目录（`Carpking_LatentUpscale`）放到 ComfyUI 的 `custom_nodes/` 下。
2. **重启 ComfyUI**。
3. 画布双击搜索 `DragonBro` 即可找到两个节点。

依赖：ComfyUI 内置 torch、comfy.utils、comfy.model_management，无需额外安装。

---

## 三、节点说明

### 1. LatentUpscaleByEnhanced（方案 A：保形放大）

**适用场景**：希望在放大后**人物/主体与原图一致**，只是变清晰。

内置 `LatentUpscaleBy` 默认用 `nearest-exact`（最近邻），整数倍放大时会做坐标取整，产生约 0.5 像素的整体错位和硬边。本节点**默认改用 `bislerp`**（latent 专用的球面线性插值），平滑且少假象。

**参数**：

| 参数 | 说明 |
| --- | --- |
| `samples` | 输入的 LATENT（图片 4 维 `B,C,H,W` / 视频 5 维 `B,C,T,H,W` 均可） |
| `upscale_by` | 放大系数，默认 2.0，范围 1.0 ~ 8.0 |
| `upscale_method` | `bislerp`(默认) / `bicubic` / `area` / `bilinear` / `nearest-exact` |

**推荐用法（保形）**：`LatentUpscaleByEnhanced(scale_by=2, bislerp)` → 接一个 `KSampler`（**denoise 0.25 ~ 0.35**，越低越保形），复用原 prompt 与 seed。全程不离开 latent 空间，人物漂移最小。

---

### 2. LatentHiresReencode（方案 B：像素级高清）

**适用场景**：追求**像素级清晰**（可接 4x 超分模型），能接受轻微结构变化。

**流程**：`latent → VAE Decode(像素) → 像素放大(插值或超分模型) → VAE Encode(回 latent)`。

**参数**：

| 参数 | 说明 |
| --- | --- |
| `samples` | 输入 LATENT（图片 / 视频均可） |
| `vae` | VAE，支持 qwen_image_vae / WanVAE 等 |
| `scale_by` | 最终放大倍数（**接超分模型时也生效**：超分模型先放大增强细节，再插值到目标尺寸） |
| `pixel_upscale_method` | 无超分模型时的像素插值方式，默认 `bicubic` |
| `upscale_model`（可选） | `UPSCALE_MODEL`，如 4x-UltraSharp / 4x-NomosWebPhoto 等 |

**省显存**：解码输出立即搬 CPU，像素放大 / 超分在 CPU 上进行，避免大像素 + encode 叠满 GPU。

> **内置修复**：ComfyUI 对 Qwen/Wan 系列 VAE 的单帧图片 latent 做 encode 时，会因时间维 crop 把 `T=1` 裁成 `T=0` 而崩溃（`UnboundLocalError`）。本节点在 encode 前临时关闭 `crop_input`、编码后恢复，已规避该 bug。

---

## 四、两条放大路线怎么选

```
        保形（人物不变）←──────────────→ 清晰（细节锐利）
  LatentUpscaleByEnhanced        折中        LatentHiresReencode
     (+低denoise refine)                 (+超分模型)
```

| 诉求 | 推荐路线 |
| --- | --- |
| 人物必须和原图一致，能接受略软 | 方案 A：`LatentUpscaleByEnhanced` + refine(denoise≈0.3) |
| 追求清晰锐利，人物轻微变化可接受 | 方案 B：`LatentHiresReencode` + 超分模型 + refine |
| 又要清楚又要保人 | 方案 A 保形 → 局部轻量超分，或 ControlNet / IPAdapter 约束后再超分 |

**关键旋钮**：放大后 refine 的 `denoise` 越低越保形（但过低会虚影/软），越高越锐利（但过高会"重新画脸"）。平衡点约 0.3 ~ 0.35。

---

## 五、兼容性与说明

- **图片 latent**（4 维 `B,C,H,W`）与**视频 latent**（5 维 `B,C,T,H,W`）均支持，空间尺寸取最后两维，时间帧自动展平/还原。
- 已针对 `qwen_image_vae`（ComfyUI 内为 WanVAE / `latent_dim=3`，单帧图片即 `T=1` 视频 latent）验证 decode→encode 往返通过。
- 超分模型走新版 ComfyUI 的 spandrel `ImageModelDescriptor` 接口（直接前向 + `tiled_scale` 分块），兼容 `Load Upscale Model`。
- 若放大后仍爆显存（小显存卡），请适当降低 `scale_by` 或输入分辨率。

---

## 六、更新记录

- **v1.x**：支持 4/5 维 latent；像素放大移至 CPU（省显存）；修复 Qwen/Wan 单帧图片 encode 的 crop bug；超分模型放大后按 `scale_by` 插值到目标尺寸。
- 早期版本：节点命名曾为 Carpking，现已改为 **DragonBro** 前缀（GUI 显示名 / 分类），代码 class_type 保持不变以保证已有工作流兼容。
