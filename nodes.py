import math
import torch
import comfy.utils
import comfy.model_management

NODE_CATEGORY = "DragonBro/Upscale"


class LatentUpscaleByEnhanced:
    """Bislerp 增强版 latent 按系数放大节点。

    内置 LatentUpscaleBy 默认用 nearest-exact（最近邻），整数倍放大时会做坐标取整，
    产生约 0.5 像素的整体错位和硬边。本节点默认改用 bislerp（latent 专用的球面线性
    插值），明显减少放大偏移；也可手动切回 bicubic / area 等其他方式。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "upscale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1}),
                "upscale_method": (
                    ["bislerp", "bicubic", "area", "bilinear", "nearest-exact"],
                    {"default": "bislerp"},
                ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = NODE_CATEGORY

    def upscale(self, samples, upscale_by, upscale_method):
        s = samples.copy()
        latent = samples["samples"]
        # 兼容图片(4维 B,C,H,W)与视频(5维 B,C,T,H,W)：空间高宽取最后两维
        H, W = latent.shape[-2], latent.shape[-1]
        width = round(W * upscale_by)
        height = round(H * upscale_by)
        if latent.ndim == 4:
            t = comfy.utils.common_upscale(latent, width, height, upscale_method, "disabled")
        else:
            B, C, T, _, _ = latent.shape
            t = latent.reshape(B * T, C, H, W)  # 时间帧并入 batch，保留通道 C
            t = comfy.utils.common_upscale(t, width, height, upscale_method, "disabled")
            t = t.reshape(B, C, T, t.shape[-2], t.shape[-1])  # 还原时间维
        s["samples"] = t
        return (s,)


class LatentHiresReencode:
    """Hires 重编码放大节点（根治 latent 直接插值的偏移）。

    流程：latent -> VAE Decode(像素) -> 像素放大(插值或超分模型) -> VAE Encode(回 latent)。

    原理：latent 是压缩的非欧特征空间，直接在 latent 上插值不忠实；而像素空间放大是
    忠实插值，重编码回 latent 会保留真实结构，偏移最小。放大后建议再接一个低噪声
    refine（denoise 0.2~0.4）补充细节。

    用法：
      - 不接超分模型：用 pixel_upscale_method（默认 bicubic）+ scale_by 做像素插值放大；
      - 接超分模型：scale_by 控制最终放大倍数（超分模型先放大增强细节，再插值到目标尺寸，
        即使目标是 2x/3x 也能用 4x 模型）。
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "vae": ("VAE",),
                "scale_by": ("FLOAT", {"default": 2.0, "min": 1.0, "max": 8.0, "step": 0.1}),
                "pixel_upscale_method": (
                    ["bicubic", "area", "bilinear", "nearest-exact"],
                    {"default": "bicubic"},
                ),
            },
            "optional": {
                "upscale_model": ("UPSCALE_MODEL",),
            },
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = NODE_CATEGORY

    def upscale(self, samples, vae, scale_by, pixel_upscale_method, upscale_model=None):
        latent = samples["samples"]
        pixels = vae.decode(latent)  # 静态图 (B,H,W,3) 或 视频 (B,T,H,W,3)
        # 省显存：解码结果立即搬到 CPU，后续像素放大/超分在 CPU 上进行，
        # 避免 decode 输出的大像素 + 放大产物 + encode 输入三者叠加占满 GPU。
        pixels = pixels.detach().cpu()

        # 兼容任意维度：空间尺寸取倒数第 3、2 维
        H, W = pixels.shape[-3], pixels.shape[-2]
        new_h = max(8, round(H * scale_by / 8) * 8)
        new_w = max(8, round(W * scale_by / 8) * 8)

        if upscale_model is not None:
            pixels = self._upscale_with_model(upscale_model, pixels)
            # 超分模型是固定倍数（如 PLKSR 为 4x），放大后再插值到 scale_by 指定的目标尺寸，
            # 这样 scale_by 能真正控制最终放大倍数；先超分增强细节、再缩放，
            # 画质比直接插值更好（即使目标是缩小到 2x，超分再缩也保留更多中高频细节）。
            pixels = self._resize_to_target(pixels, new_h, new_w)
        else:
            t = pixels.movedim(-1, -3)  # (..., C, H, W)
            orig = t.shape
            nb = math.prod(orig[:-3])  # 展平所有 leading 维度（含视频时间帧）
            Cc, Hh, Ww = orig[-3], orig[-2], orig[-1]
            align = pixel_upscale_method in ("bilinear", "bicubic")
            t = t.reshape(nb, Cc, Hh, Ww)
            t = torch.nn.functional.interpolate(
                t,
                size=(new_h, new_w),
                mode=pixel_upscale_method,
                align_corners=False if align else None,
            )
            t = t.reshape(*orig[:-3], Cc, new_h, new_w)
            pixels = t.movedim(-3, -1)  # (..., H, W, 3)

        out = samples.copy()
        # 修复 ComfyUI 单帧图片 encode 的 crop bug：
        # Qwen/Wan 图片 latent 是 T=1 的 5 维视频 latent，vae.encode 内部的
        # vae_encode_crop_pixels 会按时间下采样倍数把 T=1 裁成 T=0，
        # 导致 WanVAE.encode 收到 0 帧 -> UnboundLocalError。
        # 这里临时关闭 crop_input（像素空间尺寸已是 8 倍数，无需再裁），编码后再恢复。
        was_crop = getattr(vae, "crop_input", None)
        if was_crop is not None:
            vae.crop_input = False
        try:
            out["samples"] = vae.encode(pixels)
        finally:
            if was_crop is not None:
                vae.crop_input = was_crop
        return (out,)

    def _resize_to_target(self, pixels, new_h, new_w):
        """把超分模型放大的结果插值到 scale_by 指定的目标尺寸。

        目标尺寸与当前相同时直接返回；否则缩放（缩小用 area 更稳，放大用 bicubic）。
        """
        t = pixels.movedim(-1, -3)  # (..., C, H, W)
        orig = t.shape
        nb = math.prod(orig[:-3])  # 展平所有 leading 维度（含视频时间帧）
        Cc, Hh, Ww = orig[-3], orig[-2], orig[-1]
        if (Hh, Ww) == (new_h, new_w):
            return pixels
        mode = "area" if (Hh * Ww) > (new_h * new_w) else "bicubic"
        t = t.reshape(nb, Cc, Hh, Ww)
        t = torch.nn.functional.interpolate(
            t,
            size=(new_h, new_w),
            mode=mode,
            align_corners=None if mode == "area" else False,
        )
        t = t.reshape(*orig[:-3], Cc, new_h, new_w)
        return t.movedim(-3, -1)

    def _upscale_with_model(self, upscale_model, pixels):
        # 新版 ComfyUI 的 Load Upscale Model 加载的是 spandrel ImageModelDescriptor，
        # 直接前向调用（upscale_model(a.float())），没有老版 model_upscale 接口。
        device = comfy.model_management.get_torch_device()
        if getattr(upscale_model, "patcher", None) is not None:
            device = upscale_model.patcher.load_device
        upscale_model.to(device)

        in_img = pixels.movedim(-1, -3)  # (..., C, H, W)
        orig_shape = in_img.shape
        B = math.prod(orig_shape[:-3])  # 展平所有 leading 维度（含视频时间帧）
        C = orig_shape[-3]
        img = in_img.reshape(B, C, orig_shape[-2], orig_shape[-1]).to(device)

        scale = getattr(upscale_model, "scale", 4)
        out = comfy.utils.tiled_scale(
            img,
            lambda a: upscale_model(a.float()),
            tile_x=512,
            tile_y=512,
            overlap=32,
            upscale_amount=scale,
            output_device=comfy.model_management.intermediate_device(),
        )
        out = torch.clamp(out, min=0.0, max=1.0)
        out = out.reshape(*orig_shape[:-3], C, out.shape[-2], out.shape[-1])
        return out.movedim(-3, -1)


NODE_CLASS_MAPPINGS = {
    "LatentUpscaleByEnhanced": LatentUpscaleByEnhanced,
    "LatentHiresReencode": LatentHiresReencode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LatentUpscaleByEnhanced": "DragonBro Latent Upscale By (Enhanced)",
    "LatentHiresReencode": "DragonBro Latent Hires Reencode",
}
