import os
import gc
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import folder_paths
import comfy.utils
import comfy.samplers
import comfy.sample
import comfy.nested_tensor
import comfy.model_management as mm
import latent_preview

MMH3_LATENT_MEAN = [
    0.858090341091156, -0.9606591463088989, 1.0661640167236328, -0.5090325474739075,
    -0.2727581858634949, -1.3675414323806763, -0.2553254961967468, -0.26907554268836975,
    -0.5376840829849243, -0.0464097298681736, 0.6657370328903198, 0.19690127670764923,
    -0.5460608005523682, -0.4035342037677765, -0.23683024942874908, 0.25928452610969543,
    -0.30133944749832153, 0.211341992020607, -1.1206848621368408, 0.3581933379173279,
    -0.04225143790245056, 0.2604829967021942, 0.22864092886447906, 0.7056031823158264
]

MMH3_LATENT_STD = [
    1.2223774194717407, 1.2767263650894165, 1.6831774711608887, 1.7549455165863037,
    1.5636216402053833, 2.194143533706665, 0.9653137922286987, 1.0569885969161987,
    0.841948926448822, 0.7729952931404114, 1.8955937623977661, 0.946841835975647,
    0.7996809482574463, 0.44988900423049927, 0.7197399735450745, 0.6936293244361877,
    2.961095094680786, 2.7694199085235596, 3.0496184825897217, 2.1088054180145264,
    3.276226282119751, 3.1627357006073, 2.2816812992095947, 2.6127843856811523
]

if "latent_upscale_models" not in folder_paths.folder_names_and_paths:
    latent_upscale_dir = os.path.join(folder_paths.models_dir, "latent_upscale_models")
    os.makedirs(latent_upscale_dir, exist_ok=True)
    folder_paths.add_model_folder_path("latent_upscale_models", latent_upscale_dir)


def get_all_upscaler_safetensors():
    search_categories = ["latent_upscale_models", "upscale_models", "diffusion_models", "unet", "checkpoints"]
    found_files = []
    seen = set()

    for cat in search_categories:
        if cat in folder_paths.folder_names_and_paths:
            for filename in folder_paths.get_filename_list(cat):
                if filename.endswith((".safetensors", ".pth", ".pt", ".bin", ".ckpt")):
                    if filename not in seen:
                        seen.add(filename)
                        if any(k in filename.lower() for k in ("minimax", "h3", "upscal", "resizer")):
                            found_files.insert(0, filename)
                        else:
                            found_files.append(filename)

    found_files.append("None (Trilinear)")
    found_files.append("None (Bicubic)")
    return found_files


def resolve_safetensors_path(selected_entry: str):
    if selected_entry.startswith("None"):
        return None
    for cat in ["latent_upscale_models", "upscale_models", "diffusion_models", "unet", "checkpoints"]:
        if cat in folder_paths.folder_names_and_paths:
            full = folder_paths.get_full_path(cat, selected_entry)
            if full and os.path.isfile(full):
                return full
    if "/" in selected_entry:
        prefix, rel_name = selected_entry.split("/", 1)
        if prefix in folder_paths.folder_names_and_paths:
            full = folder_paths.get_full_path(prefix, rel_name)
            if full and os.path.isfile(full):
                return full
    return None


def clean_vram():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    mm.soft_empty_cache()


def unwrap_to_latent_dict(obj) -> dict:
    for _ in range(10):
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "args") and isinstance(obj.args, (tuple, list)) and len(obj.args) > 0:
            obj = obj.args[0]
            continue
        if isinstance(obj, (tuple, list)) and len(obj) > 0:
            obj = obj[0]
            continue
        try:
            obj = obj[0]
        except Exception:
            break
    if isinstance(obj, dict):
        return obj
    if torch.is_tensor(obj) or hasattr(obj, "tensors") or getattr(obj, "is_nested", False):
        return {"samples": obj}
    raise TypeError(f"[Kimchy Upscaler MP] Invalid latent container type: {type(obj)}")


def normalization(channels: int):
    return nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)


class ResBlockEmb3D(nn.Module):
    def __init__(self, channels: int, emb_channels: int, dropout: float = 0.0, out_channels: int = None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv3d(channels, self.out_channels, 3, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_channels, 2 * self.out_channels),
        )
        self.out_norm = normalization(self.out_channels)
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Conv3d(self.out_channels, self.out_channels, 3, padding=1),
        )
        self.skip = nn.Conv3d(channels, self.out_channels, 1) if self.out_channels != channels else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).to(dtype=h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        scale, shift = torch.chunk(emb_out, 2, dim=1)
        h = self.out_norm(h) * (1.0 + scale) + shift
        h = self.out_layers(h)
        return self.skip(x) + h


class TemporalConv(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.norm = normalization(channels)
        self.dwconv = nn.Conv3d(
            channels, channels,
            kernel_size=(kernel_size, 1, 1),
            padding=(padding, 0, 0),
            groups=channels,
        )
        self.pwconv = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.norm(x))
        h = self.pwconv(self.dwconv(h))
        return x + h


class AttnBlock3D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = normalization(channels)
        self.q = nn.Conv3d(channels, channels, 1)
        self.k = nn.Conv3d(channels, channels, 1)
        self.v = nn.Conv3d(channels, channels, 1)
        self.proj_out = nn.Conv3d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class ExactMatchedLatentResizer3D(nn.Module):
    def __init__(self, sd: dict):
        super().__init__()
        channels, in_channels = sd["conv_in.weight"].shape[:2]
        embed_dim = sd["embed.0.weight"].shape[0]

        self.conv_in = nn.Conv3d(in_channels, channels, 3, padding=1)
        self.embed = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        self.in_blocks = self._build_block_list(sd, "in_blocks", channels, embed_dim)
        self.out_blocks = self._build_block_list(sd, "out_blocks", channels, embed_dim)

        self.norm_out = normalization(channels)
        self.conv_out = nn.Conv3d(channels, in_channels, 3, padding=1)

    @staticmethod
    def _build_block_list(sd: dict, prefix: str, channels: int, embed_dim: int) -> nn.ModuleList:
        indices = sorted(
            {int(k.split(".")[1]) for k in sd.keys() if k.startswith(prefix + ".") and k.split(".")[1].isdigit()}
        )
        blocks = nn.ModuleList()
        for idx in indices:
            sub_prefix = f"{prefix}.{idx}."
            if any(k.startswith(sub_prefix + "in_layers.") for k in sd):
                blocks.append(ResBlockEmb3D(channels, embed_dim, dropout=0.0))
            elif any(k.startswith(sub_prefix + "dwconv.") for k in sd):
                k_size = sd[sub_prefix + "dwconv.weight"].shape[2]
                blocks.append(TemporalConv(channels, kernel_size=k_size))
            elif any(k.startswith(sub_prefix + "proj_out.") or k.startswith(sub_prefix + "q.") for k in sd):
                blocks.append(AttnBlock3D(channels))
            else:
                blocks.append(nn.Identity())
        return blocks

    def forward_single_window(self, x: torch.Tensor, scale: float, target_hw: tuple) -> torch.Tensor:
        new_H, new_W = target_hw
        T = x.shape[2]

        # 4K 점프 시에도 신경망 임베딩은 훈련 안전 범위(1.0~4.0) 내로 클램핑
        safe_emb_scale = max(1.0, min(4.0, float(scale)))
        scale_t = torch.tensor([safe_emb_scale - 1.0], dtype=x.dtype, device=x.device).unsqueeze(0)
        emb = self.embed(scale_t).expand(x.shape[0], -1)

        x = self.conv_in(x)
        for b in self.in_blocks:
            x = b(x, emb) if isinstance(b, ResBlockEmb3D) else b(x)

        x = F.interpolate(x, size=(T, new_H, new_W), mode="trilinear", align_corners=False)

        for b in self.out_blocks:
            x = b(x, emb) if isinstance(b, ResBlockEmb3D) else b(x)

        x = F.silu(self.norm_out(x))
        x = self.conv_out(x)
        return x

    def forward_feather_adaptive(self, x: torch.Tensor, scale: float, target_hw: tuple, overlap: int = 5) -> torch.Tensor:
        """
        1MP ~ 4K(8.3MP) 전 구간 대응:
        목표 해상도(target_hw)의 크기에 따라 시간축 윈도우 크기(window_size)를 자동으로 조절하고
        2K/4K 초대형 볼륨일 때는 누적 버퍼를 CPU에 두어 16GB GPU에서도 4K OOM을 완벽히 차단합니다.
        """
        B, C, T, H, W = x.shape
        new_H, new_W = target_hw
        dev = x.device
        dt = x.dtype

        target_latent_area = new_H * new_W
        # 1080p(120x68=8160) 이하는 33프레임, 2K(160x90=14400)는 17프레임, 4K(240x136=32640)는 9프레임으로 자동 조절
        if target_latent_area > 20000:      # 4K급
            window_size = 9
            acc_device = "cpu"
        elif target_latent_area > 10000:    # 2K급
            window_size = 17
            acc_device = "cpu"
        else:                               # 1MP ~ 1080p급
            window_size = 33
            acc_device = dev

        if T <= window_size:
            return self.forward_single_window(x, scale, target_hw)

        eff_overlap = min(overlap, window_size // 2)
        stride = max(1, window_size - eff_overlap)
        starts = list(range(0, T, stride))
        if len(starts) > 1 and starts[-1] + window_size > T:
            starts[-1] = max(0, T - window_size)
            dedup = []
            for s in starts:
                if not dedup or s != dedup[-1]:
                    dedup.append(s)
            starts = dedup

        out_acc = torch.zeros((B, C, T, new_H, new_W), dtype=torch.float32, device=acc_device)
        weight_acc = torch.zeros((1, 1, T, 1, 1), dtype=torch.float32, device=acc_device)

        pbar = comfy.utils.ProgressBar(len(starts))
        for idx, s in enumerate(starts):
            e = min(T, s + window_size)
            chunk_len = e - s
            sub_in = x[:, :, s:e, :, :].to(device=dev, dtype=dt)
            sub_out = self.forward_single_window(sub_in, scale, target_hw).to(device=acc_device, dtype=torch.float32)

            w = torch.ones(chunk_len, dtype=torch.float32, device=acc_device)
            if idx > 0 and eff_overlap > 0:
                ov = min(eff_overlap, chunk_len)
                w[:ov] = torch.linspace(0.0, 1.0, ov + 2, dtype=torch.float32, device=acc_device)[1:-1]
            if idx < len(starts) - 1 and eff_overlap > 0:
                ov = min(eff_overlap, chunk_len)
                w[-ov:] = torch.minimum(w[-ov:], torch.linspace(1.0, 0.0, ov + 2, dtype=torch.float32, device=acc_device)[1:-1])

            w_view = w.view(1, 1, chunk_len, 1, 1)
            out_acc[:, :, s:e, :, :].add_(sub_out * w_view)
            weight_acc[:, :, s:e, :, :].add_(w_view)
            del sub_in, sub_out
            if acc_device == "cpu":
                torch.cuda.empty_cache()
            pbar.update(1)

        out_acc.div_(weight_acc.clamp(min=1e-6))
        del weight_acc
        return out_acc.to(dtype=dt)


def compute_target_hw_by_mode(curr_h: int, curr_w: int, upscale_by: str, target_megapixels: float, scale: float, align: int):
    vae_stride = 16
    pixel_grid = max(16, int(align))
    latent_grid = max(1, pixel_grid // vae_stride)

    cur_px_w = float(curr_w * vae_stride)
    cur_px_h = float(curr_h * vae_stride)

    if upscale_by == "megapixels":
        # megapixels 선택 시 scale(멀티플라이어)은 완전히 무시되고 1.0MP ~ 8.5MP(4K) 목표치로 직결 계산됨
        clamped_mp = max(1.0, min(8.5, float(target_megapixels)))
        target_total_px = clamped_mp * 1_000_000.0
        aspect = cur_px_w / max(1.0, cur_px_h)
        raw_px_h = math.sqrt(target_total_px / aspect)
        raw_px_w = raw_px_h * aspect
    else:
        raw_px_w = cur_px_w * float(scale)
        raw_px_h = cur_px_h * float(scale)

    new_px_w = max(pixel_grid, int(round(raw_px_w / pixel_grid)) * pixel_grid)
    new_px_h = max(pixel_grid, int(round(raw_px_h / pixel_grid)) * pixel_grid)

    new_H = max(latent_grid, (new_px_h // vae_stride // latent_grid) * latent_grid)
    new_W = max(latent_grid, (new_px_w // vae_stride // latent_grid) * latent_grid)
    eff_scale = math.sqrt((new_H * new_W) / max(1.0, float(curr_h * curr_w)))

    return new_H, new_W, eff_scale


def run_standalone_3d_latent_upscale_mp(
    video_tensor: torch.Tensor,
    model_name: str,
    upscale_by: str,
    target_megapixels: float,
    scale: float,
    align: int,
    precision: str,
) -> torch.Tensor:
    device = mm.get_torch_device()
    target_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}.get(precision, torch.float16)
    orig_dtype = video_tensor.dtype
    orig_device = video_tensor.device

    is_4d = (video_tensor.dim() == 4)
    if is_4d:
        video_tensor = video_tensor.unsqueeze(2)

    B, C, T, H, W = video_tensor.shape
    new_H, new_W, eff_scale = compute_target_hw_by_mode(H, W, upscale_by, target_megapixels, scale, align)

    print(f"[KimchyH3-MP] Mode={upscale_by} | Latent {W}x{H} -> {new_W}x{new_H} | Pixels {new_W*16}x{new_H*16} ({(new_W*16*new_H*16)/1e6:.2f} MP) | eff_scale={eff_scale:.3f}")

    if (new_H, new_W) == (H, W):
        return video_tensor.squeeze(2) if is_4d else video_tensor

    ckpt_path = resolve_safetensors_path(model_name)

    if ckpt_path is None:
        with torch.no_grad():
            x_gpu = video_tensor.to(device=device, dtype=target_dtype)
            interp_mode = "bicubic" if "bicubic" in model_name.lower() else "bilinear"
            out = torch.empty((B, C, T, new_H, new_W), dtype=orig_dtype, device="cpu")
            for t in range(T):
                f_up = F.interpolate(x_gpu[:, :, t], size=(new_H, new_W), mode=interp_mode, align_corners=False)
                out[:, :, t] = f_up.to(device="cpu", dtype=orig_dtype)
            del x_gpu
            clean_vram()
        return (out.squeeze(2) if is_4d else out).to(orig_device)

    raw_sd = comfy.utils.load_torch_file(ckpt_path, safe_load=True)
    sd = {}
    for k, v in raw_sd.items():
        if k in ("latents_mean", "latents_std"):
            continue
        clean_k = k[len("upscaler."):] if k.startswith("upscaler.") else k
        if v.dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            v = v.to(target_dtype)
        sd[clean_k] = v
    del raw_sd

    upscaler = ExactMatchedLatentResizer3D(sd)
    missing, unexpected = upscaler.load_state_dict(sd, strict=False)
    del sd
    if missing:
        print(f"[KimchyH3-MP] Missing keys: {missing}")

    upscaler.eval().to(device=device, dtype=target_dtype)

    with torch.no_grad():
        # 4K 업스케일 시 입력 텐서는 필요한 청크만 GPU로 올리도록 처리
        x = video_tensor.to(dtype=target_dtype)
        if C == 24:
            mean = torch.tensor(MMH3_LATENT_MEAN, dtype=target_dtype, device=x.device).view(1, 24, 1, 1, 1)
            std = torch.tensor(MMH3_LATENT_STD, dtype=target_dtype, device=x.device).view(1, 24, 1, 1, 1)
            x = (x - mean) / std

        if (new_H * new_W) <= 10000:
            x = x.to(device=device)

        x = upscaler.forward_feather_adaptive(x, scale=eff_scale, target_hw=(new_H, new_W), overlap=5)

        if C == 24:
            mean_out = torch.tensor(MMH3_LATENT_MEAN, dtype=x.dtype, device=x.device).view(1, 24, 1, 1, 1)
            std_out = torch.tensor(MMH3_LATENT_STD, dtype=x.dtype, device=x.device).view(1, 24, 1, 1, 1)
            x = x * std_out + mean_out
            del mean, std, mean_out, std_out

        out_cpu = x.to(device=orig_device, dtype=orig_dtype)
        del x

    upscaler.to("cpu")
    del upscaler
    clean_vram()

    return out_cpu.squeeze(2) if is_4d else out_cpu


class KimchyBasicGuider(comfy.samplers.CFGGuider):
    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})

    def __call__(self, *args, **kwargs):
        return self.predict_noise(*args, **kwargs)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        return comfy.samplers.calc_cond_batch(
            self.inner_model, [self.conds.get("positive", None)], x, timestep, model_options
        )[0]


def unpack_mmh3_latent(latent_dict: dict):
    latent_dict = unwrap_to_latent_dict(latent_dict)
    samples = latent_dict["samples"]
    extra_meta = {k: v for k, v in latent_dict.items() if k != "samples"}

    if getattr(samples, "is_nested", False) or hasattr(samples, "tensors"):
        tensors = samples.unbind() if hasattr(samples, "unbind") else samples.tensors
        video_tensor = tensors[0].detach().to("cpu")
        audio_tensors = [t.detach().to("cpu") for t in tensors[1:]]
        return video_tensor, ("nested_tensor", (type(samples), audio_tensors), extra_meta)

    if isinstance(samples, dict) and "video" in samples:
        v = samples["video"].detach().to("cpu")
        a = samples.get("audio", None)
        return v, ("dict", a.detach().to("cpu") if torch.is_tensor(a) else a, extra_meta)

    if torch.is_tensor(samples) and samples.shape[1] > 24:
        return samples[:, :24].detach().to("cpu"), ("cat_channel", samples[:, 24:].detach().to("cpu"), extra_meta)

    return samples.detach().to("cpu"), ("pure_video", None, extra_meta)


def repack_mmh3_latent(video_tensor: torch.Tensor, pack_info: tuple) -> dict:
    mode, audio_data, extra_meta = pack_info
    out_dict = dict(extra_meta)

    if mode == "nested_tensor":
        cls_type, audio_tensors = audio_data
        restored_audios = [t.to(device=video_tensor.device, dtype=video_tensor.dtype) for t in audio_tensors]
        out_dict["samples"] = cls_type([video_tensor, *restored_audios])
    elif mode == "dict":
        res = {"video": video_tensor}
        if audio_data is not None:
            res["audio"] = audio_data.to(video_tensor.device) if torch.is_tensor(audio_data) else audio_data
        out_dict["samples"] = res
    elif mode == "cat_channel":
        if audio_data.shape[-2:] == video_tensor.shape[-2:]:
            out_dict["samples"] = torch.cat([video_tensor, audio_data.to(video_tensor.device)], dim=1)
        else:
            out_dict["samples"] = comfy.nested_tensor.NestedTensor([video_tensor, audio_data.to(video_tensor.device)])
    else:
        out_dict["samples"] = video_tensor
    return out_dict


def sync_and_crop_conditioning(conditioning, full_lat_h: int, full_lat_w: int, y0: int = None, y1: int = None, x0: int = None, x1: int = None):
    if conditioning is None:
        return None
    new_cond = []
    for item in conditioning:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            cond_tensor = item[0]
            cond_dict = dict(item[1])

            if "minimax_keyframes" in cond_dict and torch.is_tensor(cond_dict["minimax_keyframes"]):
                kf = cond_dict["minimax_keyframes"]
                if kf.shape[-2:] != (full_lat_h, full_lat_w):
                    with torch.no_grad():
                        if kf.dim() == 5:
                            B, C, T, _, _ = kf.shape
                            out_kf = torch.empty((B, C, T, full_lat_h, full_lat_w), dtype=kf.dtype, device=kf.device)
                            for t in range(T):
                                out_kf[:, :, t] = F.interpolate(kf[:, :, t], size=(full_lat_h, full_lat_w), mode="bilinear", align_corners=False)
                            kf = out_kf
                        elif kf.dim() == 4:
                            kf = F.interpolate(kf, size=(full_lat_h, full_lat_w), mode="bilinear", align_corners=False)
                if y0 is not None and x0 is not None:
                    kf = kf[..., y0:y1, x0:x1]
                cond_dict["minimax_keyframes"] = kf

            new_cond.append([cond_tensor, cond_dict])
        else:
            new_cond.append(item)
    return new_cond


def make_2d_cosine_feather_mask(h: int, w: int, top_ov: int, bot_ov: int, left_ov: int, right_ov: int, device):
    wy = torch.ones(h, dtype=torch.float32, device=device)
    wx = torch.ones(w, dtype=torch.float32, device=device)

    if top_ov > 0:
        ov = min(top_ov, h)
        wy[:ov] = 0.5 * (1.0 - torch.cos(torch.linspace(0.0, math.pi, ov + 2, device=device)[1:-1]))
    if bot_ov > 0:
        ov = min(bot_ov, h)
        wy[-ov:] = torch.minimum(wy[-ov:], 0.5 * (1.0 - torch.cos(torch.linspace(math.pi, 0.0, ov + 2, device=device)[1:-1])))
    if left_ov > 0:
        ov = min(left_ov, w)
        wx[:ov] = 0.5 * (1.0 - torch.cos(torch.linspace(0.0, math.pi, ov + 2, device=device)[1:-1]))
    if right_ov > 0:
        ov = min(right_ov, w)
        wx[-ov:] = torch.minimum(wx[-ov:], 0.5 * (1.0 - torch.cos(torch.linspace(math.pi, 0.0, ov + 2, device=device)[1:-1])))

    return (wy[:, None] * wx[None, :]).clamp(min=1e-4).view(1, 1, 1, h, w)


def compute_window_coords(total_len: int, win_len: int, overlap: int):
    if total_len <= win_len:
        return [0]
    stride = max(2, ((win_len - overlap) // 2) * 2)
    coords = list(range(0, total_len - win_len, stride))
    last = total_len - win_len
    if not coords or coords[-1] != last:
        coords.append(last)
    return coords


def apply_safe_video_color_match(video_tensor: torch.Tensor, ref_video_cpu: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        orig_dtype = video_tensor.dtype
        t_f32 = video_tensor.to(torch.float32)
        r_f32 = ref_video_cpu.to(device=video_tensor.device, dtype=torch.float32)
        dims = (2, 3, 4) if t_f32.dim() == 5 else (2, 3)
        t_mean = t_f32.mean(dim=dims, keepdim=True)
        t_std = t_f32.std(dim=dims, keepdim=True).clamp(min=1e-5)
        r_mean = r_f32.mean(dim=dims, keepdim=True)
        r_std = r_f32.std(dim=dims, keepdim=True).clamp(min=1e-5)
        ratio = (r_std / t_std).clamp(0.5, 2.0)
        return ((t_f32 - t_mean) * ratio + r_mean).to(orig_dtype)


# ==============================================================================
# 사이드 버전 노드: KimchyH3ZeroSplitMP (1.0MP ~ 4K 8.5MP 완벽 지원 & 위젯 자동 토글)
# ==============================================================================
class KimchyH3ZeroSplitMP:
    @classmethod
    def INPUT_TYPES(cls):
        safetensors_list = get_all_upscaler_safetensors()
        return {
            "required": {
                "model": ("MODEL",),
                "conditioning": ("CONDITIONING",),
                "latent": ("LATENT",),
                "noise": ("NOISE",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "model_name": (safetensors_list,),
                "upscale_by": (["megapixels", "scale_multiplier"], {"default": "megapixels"}),
                "target_megapixels": ("FLOAT", {
                    "default": 2.00, "min": 1.00, "max": 8.50, "step": 0.05,
                    "tooltip": "1.0MP(1376x768) ~ 2.0MP(1920x1088 FHD) ~ 3.7MP(2560x1440 2K) ~ 8.35MP(3840x2176 4K UHD)까지 지정 가능합니다."
                }),
                "scale": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 6.0, "step": 0.05,
                    "tooltip": "upscale_by를 scale_multiplier로 선택했을 때만 활성화됩니다."
                }),
                "align": ("INT", {"default": 32, "min": 16, "max": 128, "step": 16}),
                "precision": (["fp16", "bf16", "fp32"], {"default": "fp16"}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "color_match": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "negative": ("CONDITIONING",),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "execute_mp"
    CATEGORY = "video/KimchyH3"

    def _run_single_guider_pass(self, model, pos_cond, neg_cond, cfg, sub_latent_dict, sub_noise_tensor, sampler, sigmas, active_seed):
        if neg_cond is None and abs(cfg - 1.0) <= 1e-4:
            guider = KimchyBasicGuider(model)
            guider.set_conds(pos_cond)
        else:
            eff_neg = neg_cond if neg_cond is not None else [[torch.zeros_like(pos_cond[0][0]), dict(pos_cond[0][1])]]
            guider = comfy.samplers.CFGGuider(model)
            guider.set_conds(pos_cond, eff_neg)
            guider.set_cfg(cfg)

        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, sub_latent_dict["samples"])
        noise_mask = sub_latent_dict.get("noise_mask", None)
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        out = guider.sample(
            sub_noise_tensor,
            latent_image,
            sampler,
            sigmas,
            denoise_mask=noise_mask,
            callback=callback,
            disable_pbar=disable_pbar,
            seed=active_seed,
        )
        del guider
        return out

    def execute_mp(
        self,
        model,
        conditioning,
        latent,
        noise,
        sampler,
        sigmas,
        model_name,
        upscale_by,
        target_megapixels,
        scale,
        align,
        precision,
        cfg,
        color_match,
        negative=None,
    ):
        mm.unload_all_models()
        clean_vram()

        video_tensor, pack_info = unpack_mmh3_latent(latent)

        up_video_tensor = run_standalone_3d_latent_upscale_mp(
            video_tensor=video_tensor,
            model_name=model_name,
            upscale_by=upscale_by,
            target_megapixels=target_megapixels,
            scale=scale,
            align=align,
            precision=precision,
        )
        del video_tensor

        new_H, new_W = up_video_tensor.shape[-2], up_video_tensor.shape[-1]
        ref_upscaled_video_cpu = up_video_tensor.detach().to("cpu").clone() if color_match else None

        if sigmas is None or sigmas.numel() <= 1:
            return (repack_mmh3_latent(up_video_tensor, pack_info),)

        full_upscaled_dict = repack_mmh3_latent(up_video_tensor, pack_info)
        full_noise = noise.generate_noise(full_upscaled_dict)
        full_noise_v, _ = unpack_mmh3_latent({"samples": full_noise})
        active_seed = getattr(noise, "noise_seed", getattr(noise, "seed", 0))

        win_h = min(new_H, 44)
        win_w = min(new_W, 56)
        ov_h = min(12, win_h // 3)
        ov_w = min(16, win_w // 3)

        y_coords = compute_window_coords(new_H, win_h, ov_h)
        x_coords = compute_window_coords(new_W, win_w, ov_w)

        if len(y_coords) == 1 and len(x_coords) == 1:
            pos_c = sync_and_crop_conditioning(conditioning, new_H, new_W)
            neg_c = sync_and_crop_conditioning(negative, new_H, new_W) if negative is not None else None
            samples_out = self._run_single_guider_pass(
                model, pos_c, neg_c, cfg, full_upscaled_dict, full_noise, sampler, sigmas, active_seed
            )
            sampled_v, _ = unpack_mmh3_latent({"samples": samples_out})
        else:
            B, C, T, _, _ = up_video_tensor.shape
            v_acc = torch.zeros((B, C, T, new_H, new_W), dtype=torch.float32, device="cpu")
            w_acc = torch.zeros((1, 1, 1, new_H, new_W), dtype=torch.float32, device="cpu")

            for yi, y0 in enumerate(y_coords):
                y1 = y0 + win_h
                top_ov = (y_coords[yi - 1] + win_h - y0) if yi > 0 else 0
                bot_ov = (y1 - y_coords[yi + 1]) if yi < len(y_coords) - 1 else 0

                for xi, x0 in enumerate(x_coords):
                    x1 = x0 + win_w
                    left_ov = (x_coords[xi - 1] + win_w - x0) if xi > 0 else 0
                    right_ov = (x1 - x_coords[xi + 1]) if xi < len(x_coords) - 1 else 0

                    sub_v = up_video_tensor[:, :, :, y0:y1, x0:x1].contiguous()
                    sub_n_v = full_noise_v[:, :, :, y0:y1, x0:x1].contiguous()

                    sub_latent_dict = repack_mmh3_latent(sub_v, pack_info)
                    sub_noise_dict = repack_mmh3_latent(sub_n_v, pack_info)

                    pos_c = sync_and_crop_conditioning(conditioning, new_H, new_W, y0, y1, x0, x1)
                    neg_c = sync_and_crop_conditioning(negative, new_H, new_W, y0, y1, x0, x1) if negative is not None else None

                    sub_out = self._run_single_guider_pass(
                        model, pos_c, neg_c, cfg, sub_latent_dict, sub_noise_dict["samples"], sampler, sigmas, active_seed
                    )
                    sub_out_v, _ = unpack_mmh3_latent({"samples": sub_out})

                    if color_match:
                        sub_ref = ref_upscaled_video_cpu[:, :, :, y0:y1, x0:x1]
                        sub_out_v = apply_safe_video_color_match(sub_out_v, sub_ref)

                    w_mask = make_2d_cosine_feather_mask(win_h, win_w, top_ov, bot_ov, left_ov, right_ov, "cpu")
                    v_acc[:, :, :, y0:y1, x0:x1] += sub_out_v.to(torch.float32) * w_mask
                    w_acc[:, :, :, y0:y1, x0:x1] += w_mask

                    del sub_v, sub_n_v, sub_out, sub_out_v
                    clean_vram()

            sampled_v = (v_acc / w_acc.clamp(min=1e-6)).to(dtype=up_video_tensor.dtype)
            del v_acc, w_acc

        del up_video_tensor, full_noise, full_noise_v
        if color_match and ref_upscaled_video_cpu is not None:
            sampled_v = apply_safe_video_color_match(sampled_v, ref_upscaled_video_cpu)
            del ref_upscaled_video_cpu

        out_latent_dict = repack_mmh3_latent(sampled_v.to(mm.intermediate_device()), pack_info)
        clean_vram()
        return (out_latent_dict,)
