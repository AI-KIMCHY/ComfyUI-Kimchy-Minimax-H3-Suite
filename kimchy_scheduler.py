import math
import torch
import comfy.samplers


class KimchyH3DualScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "schedule_mode": (
                    ["h3_shifted_simple", "h3_uniform_logsnr", "beta", "normal"],
                    {"default": "h3_shifted_simple"}
                ),
                "total_steps": ("INT", {
                    "default": 9, "min": 2, "max": 100, "step": 1,
                    "tooltip": "전체 스텝 수 (4스텝 로라=5, 8스텝 로라=9, 순정 모델=20~30)"
                }),
                "split_mode": (
                    ["keep_stage2_steps (권장: 5~30스텝 전자동)", "split_by_sigma_threshold (시그마 0.60 기준 자동분할)", "manual_stage1_steps (기존 수동 방식)"],
                    {"default": "keep_stage2_steps (권장: 5~30스텝 전자동)"}
                ),
                "stage2_refine_steps": ("INT", {
                    "default": 1, "min": 1, "max": 10, "step": 1,
                    "tooltip": "2차 고해상도 업스케일러(low_sigmas)에 할당할 스텝 수입니다."
                }),
                "manual_stage1_steps": ("INT", {
                    "default": 8, "min": 1, "max": 99, "step": 1,
                    "tooltip": "split_mode가 manual_stage1_steps일 때만 사용되는 1차 스텝 수입니다."
                }),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01}),
                "auto_balance_stage2_sigma": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "4스텝(total=5)이나 순정(total=30)으로 바꿨을 때도 2차 시작 시그마가 황금 범위(0.45~0.60)를 유지하도록 자동 보정합니다."
                }),
            }
        }

    RETURN_TYPES = ("SIGMAS", "SIGMAS")
    RETURN_NAMES = ("high_sigmas", "low_sigmas")
    FUNCTION = "generate_dual_sigmas"
    CATEGORY = "video/KimchyH3"

    def _build_full_sigmas(self, model, schedule_mode: str, total_steps: int) -> torch.Tensor:
        model_sampling = model.get_model_object("model_sampling")

        if schedule_mode in ("beta", "normal"):
            return comfy.samplers.calculate_sigmas(model_sampling, schedule_mode, total_steps).cpu().to(torch.float32)

        base_sigmas = comfy.samplers.calculate_sigmas(model_sampling, "simple", total_steps).cpu().to(torch.float32)

        if schedule_mode == "h3_uniform_logsnr" and total_steps > 2:
            s_first = float(base_sigmas[0])
            s_second = float(base_sigmas[1])
            s_last = float(base_sigmas[-2])

            lam_start = math.log(max(1e-6, (1.0 - s_second) / max(1e-6, s_second)))
            lam_end = math.log(max(1e-6, (1.0 - s_last) / max(1e-6, s_last)))

            lams = torch.linspace(lam_start, lam_end, total_steps - 1, dtype=torch.float32)
            mid_sigmas = 1.0 / (1.0 + torch.exp(lams))

            return torch.cat([
                torch.tensor([s_first], dtype=torch.float32),
                mid_sigmas,
                torch.tensor([0.0], dtype=torch.float32)
            ])

        return base_sigmas

    def generate_dual_sigmas(
        self,
        model,
        schedule_mode: str,
        total_steps: int,
        split_mode: str,
        stage2_refine_steps: int,
        manual_stage1_steps: int,
        denoise: float,
        auto_balance_stage2_sigma: bool = True,
    ):
        eff_total = max(2, int(total_steps))
        if denoise < 1.0:
            expanded_steps = int(round(eff_total / max(1e-4, denoise)))
            full_sigmas = self._build_full_sigmas(model, schedule_mode, expanded_steps)
            full_sigmas = full_sigmas[-(eff_total + 1):]
        else:
            full_sigmas = self._build_full_sigmas(model, schedule_mode, eff_total)

        num_transitions = len(full_sigmas) - 1

        if split_mode.startswith("keep_stage2_steps"):
            s2_steps = max(1, min(num_transitions - 1, int(stage2_refine_steps)))
            split_idx = num_transitions - s2_steps
        elif split_mode.startswith("split_by_sigma_threshold"):
            valid_sigmas = full_sigmas[:-1]
            diffs = torch.abs(valid_sigmas - 0.60)
            best_idx = int(torch.argmin(diffs).item())
            split_idx = max(1, min(num_transitions - 1, best_idx))
        else:
            split_idx = max(1, min(num_transitions - 1, int(manual_stage1_steps)))

        high_sigmas = full_sigmas[:split_idx + 1].clone()
        low_sigmas = full_sigmas[split_idx:].clone()

        if auto_balance_stage2_sigma and low_sigmas.numel() >= 2:
            orig_start = float(low_sigmas[0])
            target_start = None
            if orig_start > 0.6001:
                target_start = 0.60
            elif orig_start < 0.4500 and low_sigmas.numel() == 2:
                target_start = 0.50

            if target_start is not None and orig_start > 1e-5:
                scale_ratio = target_start / orig_start
                low_sigmas[:-1] = (low_sigmas[:-1] * scale_ratio).clamp(min=1e-4, max=0.999)
                low_sigmas[-1] = 0.0

        high_sigmas = high_sigmas.clamp(0.0, 1.0)
        low_sigmas = low_sigmas.clamp(0.0, 1.0)

        return (high_sigmas, low_sigmas)
