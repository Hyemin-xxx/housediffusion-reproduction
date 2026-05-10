# HouseDiffusion Reproduction on Apple Silicon (M1)

> **CVPR 2023 논문 [HouseDiffusion](https://arxiv.org/abs/2211.13287) 의 추론 파이프라인을 MacBook Air M1 환경에서 재현한 기록입니다.**
> 성균관대학교 스마트팩토리융합학과 / 발표자: Hyemin Kim

원본 논문: Shabani et al., *HouseDiffusion: Vector Floorplan Generation via a Diffusion Model with Discrete and Continuous Denoising*, CVPR 2023
원본 코드: https://github.com/aminshabani/house_diffusion

---

## SUMMARY

원본 코드는 NVIDIA CUDA 전용으로, Apple Silicon에서 그대로 동작하지 않습니다. 본 저장소는 **CUDA → MPS 포팅 + dtype 호환성 수정 + 데이터 로더 수정**을 통해 M1에서 추론 파이프라인을 end-to-end 동작시킨 결과물입니다.

**결과물 품질에 대한 정직한 고지**: RPLAN 데이터셋(라이센스 신청 후 1주일째 미수령)을 받지 못해, MagicPlan에서 변환된 임시 데이터(라벨 매핑 미완성)로 추론을 실행했습니다. **1000-step diffusion은 완료되지만 시각적으로 의미있는 평면도는 생성되지 않습니다.** 이는 데이터 도메인 차이에 의한 것이며, 자세한 분석은 아래 [결과 분석](#결과-분석) 섹션에 있습니다.

---

## 실행 환경

| 항목 | 논문 원본 | 본 재현 |
|---|---|---|
| GPU | NVIDIA RTX 6000 (24GB) | Apple M1 (MPS) |
| OS | Linux | macOS |
| Python | 3.8 | 3.9 |
| PyTorch | CUDA build | MPS build |
| 분산 학습 | mpi4py | 단일 프로세스 |

---

## 빠른 시작

```bash
# 1. 환경 생성
conda create -n housediffusion python=3.9 -y
conda activate housediffusion

# 2. 의존성 설치
pip install torch torchvision  # MPS build (PyTorch 2.0+)
pip install -r requirements.txt

# 3. 사전학습 모델 (model250000.pt) 다운로드 후 ckpts/exp/ 에 배치
# (원본 저장소 README 참조)

# 4. 데이터셋 전처리 (학습 통계 생성)
mkdir -p processed_rplan
python -c "
from house_diffusion.rplanhg_datasets import RPlanhgDataset
RPlanhgDataset(set_name='train', analog_bit=False, target_set=8)
"

# 5. 추론 실행
python scripts/image_sample.py \
    --dataset rplan \
    --batch_size 1 \
    --set_name eval \
    --target_set 8 \
    --num_samples 4 \
    --model_path ckpts/exp/model250000.pt

# 6. 결과 시각화 (npz → PNG)
python scripts/visualize_samples.py --target_set 8
open outputs/set8/png/
```

---

## 원본 코드 대비 수정 사항

### 1. `scripts/image_sample.py` — 전체 재작성

**원본 문제점**
- `dist_util.dev()` (분산 학습 디바이스 선택) 호출
- numpy object dtype 배열 → tensor 변환 실패
- `torch.distributed.init_process_group` 초기화 (macOS에서 깨짐)

**수정 내용**
- `pick_device()`: MPS → CUDA → CPU 우선순위 자동 선택
- `to_device_recursive()`: `dtype=object` numpy 배열을 안전하게 numeric으로 풀어내는 변환기 신규 작성
- 분산 학습 코드 제거 (단일 프로세스)
- fp16 강제 비활성 (MPS 안전성)
- 연속 실패 N회 시 즉시 중단 (무한루프 방지)

### 2. `house_diffusion/gaussian_diffusion.py` 976행

**원본**
```python
res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
```

**수정**
```python
res = th.from_numpy(arr).float().to(device=timesteps.device)[timesteps]
```

`.float()` 호출 시점을 `.to(device)` 앞으로 이동. MPS는 float64 텐서를 받지 못하므로 CPU에서 먼저 float32로 변환 후 전송.

### 3. `house_diffusion/rplanhg_datasets.py` 289행 근처

**원본**
```python
for i, room in enumerate(h):
    if room[1]==1:
        living_room_index = i
        break
```

거실(type 1)이 없는 데이터에서 `UnboundLocalError` 발생.

**수정**: `for` 루프 위에 `living_room_index = 0` fallback 추가.

### 4. 환경 변수

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
```

`scripts/image_sample.py` 상단에서 자동 설정됨.

### 5. `base_dir` 경로 수정

`rplanhg_datasets.py` 89행: `'../datasets/rplan'` → `'datasets/rplan'`

---

## 실행 결과

### 실행 로그 (MacBook Air M1, 16GB)

```
[M1] using device: mps
creating model and diffusion...
Number of model parameters: 26541330
COSINE
creating data loader...
sampling...
loading eval of target set 8
1000it [00:48, 20.82it/s]
  saved 29/4
done. outputs at outputs/set8
```

- **성능**: 1000-step denoising이 약 48초/샘플 (≈ 20.82 it/s)
- **메모리**: M1 통합 메모리 16GB로 batch_size=1 구동 가능
- **출력**: `outputs/set8/sample_*.npz` 형식으로 저장

### 저장된 데이터 형상

| 항목 | 값 |
|---|---|
| sample.shape | (29, 1, 2, 100) — 모든 timestep |
| 좌표 범위 | [-0.91, 0.88] — 정규화 정상 |
| 평균 / 표준편차 | -0.04 / 0.34 |
| 실제 코너 수 | 78개 (padding 제외) |

---

## 결과 분석

### 관찰

생성된 78개 코너가 **단일 방(type=0, Unknown)** 으로 묶여 출력되었습니다.

| 항목 | 정상 RPLAN | 본인 결과 |
|---|---|---|
| 방 개수 | 5~8개 | **1개** |
| 방당 코너 | 4~8개 | **78개** |
| 방 종류 | Living/Bed/Kitchen/Bath/... | **Unknown 한 종류** |

### 근본 원인: 데이터 라벨링 미완성

본인 데이터의 room_type 분포:

```
type 1 (Living):   34회
type 2 (Kitchen):  34회
type 3 (Bedroom):  34회
type 4 (Bathroom): 34회
type 5 (Balcony):  34회
type 6 (Entrance): 34회
type 7 (Dining):   34회
type 8 (Study):   486회   ← 이상하게 많음
```

100개 파일 중 **66개 파일은 모든 방이 type 8로 찍혀 있고**, 나머지 34개에만 type 1~7이 균일하게 한 번씩 들어가 있습니다. 이는 변환 스크립트가 라벨 매핑을 제대로 하지 못하고 기본값 8로 채워 넣었음을 의미합니다.

모델은 RPLAN 60,000개 평면도(거실 1 + 침실 + 주방 + 화장실 등의 일반적 조합)로 학습되었으므로, **학습 분포 외 입력**에 대해 의미있는 평면도를 생성할 수 없습니다.

### 결론

- ✅ **1000-step diffusion은 정상 완료** (좌표가 [-1, 1] 범위에 잘 정렬됨)
- ✅ **코드 파이프라인 자체는 동작** (M1/MPS 포팅 성공)
- ❌ **결과 시각화는 노이즈 형상** — 데이터 라벨 부재로 모델이 의미있는 구조를 만들지 못함

논문 Figure 1과 비교하면, 본인 결과는 t=1000(가장 왼쪽, 노이즈) 단계와 시각적으로 유사하나 실제로는 t=0(완료) 시점의 출력입니다. 좌표 통계는 정상 종료를 보여줍니다.

---

## 저장소 구조

```
housediffusion-reproduction/
├── house_diffusion/         # 원본 코드 + 수정사항
│   ├── house_diffusion/
│   │   ├── gaussian_diffusion.py  ← 976행 수정
│   │   ├── rplanhg_datasets.py    ← 289행 수정 + base_dir
│   │   └── ...
│   ├── scripts/
│   │   ├── image_sample.py        ← 전체 재작성
│   │   └── visualize_samples.py   ← 신규
│   └── ckpts/exp/
│       └── model250000.pt         (다운로드 필요)
├── datasets/rplan/                (학습/평가 데이터, 별도 구성)
└── README.md                      (이 파일)
```

---

## 배운 점

1. **벤더 락인 코드 포팅의 핵심은 dtype 호환성** — CUDA→MPS는 디바이스 교체보다 float64↔float32 처리가 훨씬 까다롭다.
2. **데이터 품질이 모델보다 우선** — 사전학습 모델도 도메인 밖 입력에는 의미있는 결과를 내지 못한다.
3. **재현 연구의 가치는 "왜 안 됐는지"의 명확한 기록** — 실패한 케이스의 원인 분석은 다음 연구자에게 가장 도움 되는 자료다.

---

## 향후 계획

- **단기**: RPLAN 라이센스 도착 시 동일 파이프라인으로 정식 재현 (코드 변경 불필요)
- **중기**: MagicPlan 원본 도착 시 라벨 매핑 검증된 변환기 작성
- **장기 (캡스톤 연계)**: GMP 시설 도면 데이터셋으로 변환, 방 종류 vocabulary를 Grade A/B/C/D + Airlock + CNC 등으로 교체하여 fine-tuning

---

## 라이센스

원본 코드는 [원본 저장소 라이센스](https://github.com/aminshabani/house_diffusion) 를 따릅니다. 본 재현 저장소의 수정 사항은 동일 라이센스 하에 공개합니다.

## Acknowledgement

원본 논문 저자: Mohammad Amin Shabani, Sepidehsadat Hosseini, Yasutaka Furukawa (Simon Fraser University)
