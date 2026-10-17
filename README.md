# 🌶️ ComfyUI-Kimchy-H3-Suite

**MiniMax H3 (FLOW_AV)** 비디오·오디오 동시 생성 모델을 위한 **무결점·저VRAM 올인원 3D 잠재 업스케일러(`Kimchy H3 Upscaler V2` / `Side-MP 1MP~4K`)** 및 **전천후 듀얼 스케줄러(`Kimchy H3 Dual Scheduler`)** 통합 패키지입니다.

기존의 복잡한 3단 분리 노드 구조(`3D Latent Upscaler` + `Spatial Split Params` + `Split Upscale`)와 스케줄러 분리 구조(`BasicScheduler` + `SplitSigmas`)를 각각 **단 하나의 통합 노드**로 압축하여, 타일 경계선(Seam)이나 격자 오염(Checkerboard) 없이 16GB급 GPU에서도 빠르고 깨끗한 고해상도 업스케일링을 구현합니다.

---

## 📦 포함된 노드 (Included Nodes)

### 1. 🌶️ Kimchy H3 Upscaler V2 (`KimchyH3ZeroSplitV2`)
* **3-in-1 올인원 통합**: 배율(`scale`: 1.0x ~ 4.0x) 기반의 3D 신경망 잠재 공간 업스케일링과 2차 DiT 디테일 리파인(Refinement)을 단일 노드에서 한 번에 처리합니다.
* **체커보드/줄무늬 오염 원천 차단**: 체크포인트(`minimax_h3_latent_upscaler_3d_*.safetensors`)의 레이어 인덱스를 1:1 동적 매칭하고, RoPE 주파수 붕괴를 막는 **와이드 윈도우 코사인 페더링(Wide-Window Cosine Feathering)**으로 경계선 없이 융합합니다.
* **피크 VRAM 50% 절감 & 어텐션 불간섭**: 3D 업스케일 직후 가중치를 즉시 언로드(`Auto-Unload`)하며, 사용자가 설정한 어텐션 백엔드(`SageAttention`, `Comfy Kitchen` 등)를 강제로 변경하지 않고 100% 존중합니다.
* **네이티브 AV `NestedTensor` 완벽 보존**: 24채널 비디오 + 32채널 스테레오 오디오 구조를 유지하여 단일 출력을 `VAEDecode`와 `VAEDecodeAudio`에 동시에 직결할 수 있습니다.

### 2. 🌶️ Kimchy H3 Upscaler (Side-MP 1MP~4K) (`KimchyH3ZeroSplitMP`)
* **메가픽셀(`1.0 MP ~ 8.5 MP 4K UHD`) 및 배율(`scale_multiplier`) 선택 지원**: 입력 영상의 가로:세로 종횡비(Aspect Ratio)를 자동 유지한 채 원하는 목표 메가픽셀(예: `1.0MP`, `1.5MP`, `2.0MP FHD`, `3.7MP 2K`, `8.35MP 4K`)로 정확히 업스케일합니다.
* **스마트 UI 위젯 자동 토글**: `megapixels` 선택 시 불필요한 `scale` 위젯을 자동으로 숨기고, `scale_multiplier` 선택 시 `target_megapixels` 위젯을 숨깁니다.
* **4K 적응형 메모리 관리**: 2K/4K 초고해상도 타겟 시 시간축 윈도우 크기와 누적 버퍼를 자동으로 조절하여 OOM(메모리 부족)을 방지합니다.

### 3. 🌶️ Kimchy H3 Dual Scheduler (`KimchyH3DualScheduler`)
* **2-in-1 듀얼 출력 (`high_sigmas` / `low_sigmas`)**: `BasicScheduler`와 `SplitSigmas` 두 노드를 하나로 통합하여 1차 생성용과 2차 리파인용 시그마를 동시에 출력합니다.
* **4스텝 터보 ~ 30스텝 순정 100% 전자동 대응 (`keep_stage2_steps`)**: 4스텝 LoRA(`total_steps=5`), 8스텝 LoRA(`total_steps=9`), 순정 모델(`total_steps=20~30`) 등 구성을 바꿀 때 **`total_steps` 숫자 하나만 변경**하면 자동으로 `(전체 - N) : N` 비율로 1차/2차 스텝을 분리합니다.
* **2차 시그마 황금 밸런스 자동 보호 (`auto_balance_stage2_sigma`)**: 총 스텝 수를 줄이거나 늘려도 2차 시작 시그마를 황금 구간(`0.45 ~ 0.60`)으로 자동 밸런싱합니다.
* **`h3_uniform_logsnr` 전용 커브 및 범용 호환**: `MiniMaxH3Sampler`의 2차 다단계 보정에 최적화된 등간격 Log-SNR 커브를 지원하며, `euler`, `res_multistep` 등 모든 표준 ComfyUI 샘플러와도 100% 호환됩니다.

---

## 🚀 설치 방법 (Installation)

### 1. 커스텀 노드 설치
`ComfyUI/custom_nodes/` 경로에서 Git Clone 하거나 압축을 해제합니다.
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/YOUR_USERNAME/ComfyUI-Kimchy-H3-Suite.git
```

**폴더 구조:**
```text
ComfyUI/custom_nodes/ComfyUI-Kimchy-H3-Suite/
 ├── web/
 │    └── kimchy_mp_ui.js
 ├── __init__.py
 ├── kimchy_upscaler_v2.py
 ├── kimchy_upscaler_mp.py
 ├── kimchy_scheduler.py
 ├── pyproject.toml
 ├── LICENSE
 └── README.md
```

### 2. 3D Latent Upscaler 가중치 파일 배치
MiniMax H3 3D Latent Upscaler 가중치 파일(`.safetensors`)을 아래 폴더 중 편한 곳에 넣어두면 노드가 자동으로 스캔하여 드롭다운에 표시합니다:
* `ComfyUI/models/latent_upscale_models/` *(권장)*
* `ComfyUI/models/upscale_models/`

> **권장 가중치 파일**: `minimax_h3_latent_upscaler_3d_fp16.safetensors` 또는 `minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors`

---

## 🔌 워크플로우 연결 가이드 (How to Connect)

1. **`🌶️ Kimchy H3 Dual Scheduler`**
   * **입력**: `model` ← H3 본체 모델 연결
   * **출력**: `high_sigmas` → 1차 샘플러(`SamplerCustomAdvanced`)의 `sigmas` 연결 / `low_sigmas` → 김치 업스케일러의 `sigmas` 연결
2. **`🌶️ Kimchy H3 Upscaler V2` (또는 `Side-MP`)**
   * **입력**: `model`, `conditioning`, `latent`(1차 샘플러의 `denoised_output`), `noise`(`RandomNoise`), `sampler`, `sigmas`(`low_sigmas`) 연결
   * **출력**: `latent` → **`VAEDecode`**(비디오) 및 **`VAEDecodeAudio`**(오디오)의 `samples` 입력에 동시 연결

---

## ⚙️ 주요 파라미터 요약 (Parameters)

| 노드 | 옵션명 | 기본값 | 설명 |
| :--- | :--- | :---: | :--- |
| **Upscaler** | `model_name` | 자동 스캔 | 사용할 3D Latent Upscaler `.safetensors` 가중치 선택 |
| **Upscaler (MP)** | `upscale_by` | `megapixels` | `megapixels` (1.0~8.5MP 목표 지정) 또는 `scale_multiplier` (배율 지정) |
| **Upscaler (MP)** | `target_megapixels` | `2.00` | 목표 해상도 메가픽셀 (`1.0MP`=768p, `2.0MP`=1080p, `3.7MP`=2K, `8.35MP`=4K) |
| **Upscaler** | `scale` | `2.0` | 배율 모드 시 해상도 확대 배율 (`1.0` ~ `6.0`) |
| **Upscaler** | `align` | `32` | MiniMax H3 VAE(16x) 및 2×2 패치 규격에 맞춘 픽셀 정렬 단위 |
| **Upscaler** | `precision` | `fp16` | 3D 업스케일러 연산 정밀도 (`fp16` / `bf16` / `fp32`) |
| **Upscaler** | `cfg` | `1.0` | `1.0`일 때 네거티브 연산을 생략하는 고속 `BasicGuider` 작동 |
| **Upscaler** | `color_match` | `True` | 2차 리파인 후 비디오 24채널 평균/분산 드리프트 안전 보정 |
| **Scheduler** | `schedule_mode` | `h3_shifted_simple` | `h3_shifted_simple` (순정 동일), `h3_uniform_logsnr` (등간격 Log-SNR), `beta`, `normal` |
| **Scheduler** | `total_steps` | `9` | 전체 스텝 수 (4스텝 LoRA=`5`, 8스텝 LoRA=`9`, 순정=`20~30`) |
| **Scheduler** | `split_mode` | `keep_stage2_steps` | `total_steps` 변경 시에도 항상 마지막 `stage2_refine_steps`만큼만 2차에 자동 할당 |
| **Scheduler** | `stage2_refine_steps` | `1` | 2차 고해상도 업스케일 리파인에 할당할 스텝 수 |
| **Scheduler** | `auto_balance_stage2_sigma` | `True` | 스텝 수를 크게 바꾸더라도 2차 시작 시그마를 황금 구간(`0.45 ~ 0.60`)으로 자동 보호 |

---

## 📜 라이선스 (License)

이 프로젝트는 **Apache License 2.0** 라이선스에 따라 배포됩니다.
누구나 자유롭게 사용, 수정 및 배포할 수 있으나, 재배포 또는 수정본 배포 시 반드시 **원작자(Copyright) 표시를 유지**해야 하며 **수정된 사항(State changes)을 명확히 표기**해야 합니다. 자세한 내용은 [LICENSE](./LICENSE) 파일을 참고하세요.

```text
Copyright 2026 Kimchy

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
