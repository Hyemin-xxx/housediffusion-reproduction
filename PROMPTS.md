# Prompt Log — HouseDiffusion Apple Silicon 재현

> 본 문서는 본 재현 작업에 사용된 핵심 프롬프트들을 git 커밋 이력과 README 변경 사항을 토대로 **재구성한 기록**입니다. 실제 입력 원문과 토씨까지 일치하지는 않지만, 작업 단위와 의도를 보존하도록 작성되었습니다.
>
> - 사용 모델: Claude (Claude Code CLI 환경) / 일부 ChatGPT 병행
> - 작업 기간: 2026-05 ~ 2026-05
> - 작업 환경: macOS (Apple Silicon M1, 16GB) / Python 3.9 / PyTorch MPS build

---

## 0. 사전 조사

### P0-1. 논문 / 코드 출처 확인
```
HouseDiffusion (CVPR 2023) 논문 원본 코드 저장소가 어디인지 알려줘.
Apple Silicon M1에서도 동작 가능한지, CUDA 의존성이 어디에 있는지 미리 파악하고 싶어.
```

### P0-2. 재현 범위 결정
```
학습까지는 시간/자원상 불가능할 것 같고, 사전학습 체크포인트(model250000.pt)로
추론 파이프라인만 end-to-end 동작시키는 것을 1차 목표로 잡고 싶어. 합리적일까?
```

---

## 1. 환경 셋업

### P1-1. conda 환경 / 의존성
```
Python 3.9 conda env 'housediffusion'을 만들고 원본 requirements.txt 를 설치해줘.
CUDA 빌드가 들어 있으면 MPS 빌드로 바꿔서 설치해야 해.
```

### P1-2. 서브모듈 구성
```
원본 코드와 guided-diffusion 의존성을 git submodule로 추가해서 재현 저장소를 만들어줘.
저장소명은 housediffusion-reproduction.
```

### P1-3. 사전학습 모델 배치
```
원본 README의 다운로드 링크에서 model250000.pt 를 받아서
house_diffusion/ckpts/exp/ 에 배치하는 절차를 알려줘.
```

---

## 2. CUDA → MPS 포팅

### P2-1. 실행 실패 진단 (1차)
```
scripts/image_sample.py 를 실행하면 dist_util.dev() 와
torch.distributed.init_process_group 에서 죽어. macOS에서 분산학습 자체가 안 되는
환경이라서 이 부분을 단일 프로세스로 우회하고 싶어.
```

### P2-2. 디바이스 선택 로직
```
MPS → CUDA → CPU 순으로 사용 가능한 디바이스를 고르는 pick_device() 헬퍼를
image_sample.py 상단에 추가해줘. 환경변수 PYTORCH_ENABLE_MPS_FALLBACK=1 도
스크립트 안에서 자동 설정되게 해줘.
```

### P2-3. image_sample.py 전체 재작성
```
원본 image_sample.py 를 분산학습 의존성 없이, 단일 프로세스 + MPS 안전성 위주로
다시 써줘. fp16은 MPS에서 불안정하니까 강제 비활성화. 그리고 연속 N회 실패 시
무한 루프로 갇히지 않게 즉시 중단하는 가드 추가.
```

---

## 3. dtype 호환성 수정

### P3-1. float64 텐서 MPS 전송 에러
```
gaussian_diffusion.py 976행 즈음에서
'Cannot convert a MPS Tensor to float64 dtype' 비슷한 에러가 나.
원인 분석하고 최소 수정 패치 줘.
```

### P3-2. 패치 적용
```
res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
→
res = th.from_numpy(arr).float().to(device=timesteps.device)[timesteps]
.float() 호출 시점을 .to(device) 앞으로 이동.
```

### P3-3. numpy object 배열 처리
```
배치 안에 dtype=object 인 numpy 배열이 섞여 있어서 torch.tensor 변환이
실패해. dict / list / ndarray 가 섞인 구조를 재귀적으로 풀어내면서
object dtype을 numeric으로 변환하는 to_device_recursive() 헬퍼를 만들어줘.
```

---

## 4. 데이터 로더 수정

### P4-1. base_dir 경로
```
rplanhg_datasets.py 89행의 base_dir 가 '../datasets/rplan' 으로 되어 있어서
저장소 루트에서 실행하면 데이터셋을 못 찾아. './datasets/rplan' 으로 바꿔줘.
```

### P4-2. UnboundLocalError 가드
```
rplanhg_datasets.py 289행 부근에서 거실(room type==1)이 없는 평면도가 들어오면
living_room_index 가 정의되지 않은 채로 사용돼서 UnboundLocalError 가 발생해.
for 루프 위에 living_room_index = 0 으로 fallback 초기화 추가.
```

### P4-3. 학습 통계 사전 생성
```
RPlanhgDataset(set_name='train', analog_bit=False, target_set=8) 을
한 번 호출해서 processed_rplan/ 아래에 학습 통계 캐시를 만들어둬야 한다고 했는데
정확히 어떤 파일들이 만들어지는지 확인해줘.
```

---

## 5. 추론 실행

### P5-1. 첫 추론
```
python scripts/image_sample.py \
    --dataset rplan --batch_size 1 --set_name eval \
    --target_set 8 --num_samples 4 \
    --model_path ckpts/exp/model250000.pt
이렇게 실행했을 때 로그가 어떻게 나와야 정상인지 알려줘.
```

### P5-2. 실행 로그 해석
```
[M1] using device: mps
Number of model parameters: 26541330
1000it [00:48, 20.82it/s]
saved 29/4
done. outputs at outputs/set8

이렇게 떴는데 1000-step diffusion이 정상 완료된 게 맞아? "29/4" 는 무슨 의미?
```

---

## 6. 결과 시각화

### P6-1. visualize_samples.py 신규
```
outputs/set8/sample_*.npz 에는 (29, 1, 2, 100) 형상의 텐서가 들어 있어.
이걸 평면도 PNG로 그려주는 scripts/visualize_samples.py 를 새로 작성해줘.
- 좌표 범위는 [-1, 1] 정규화
- 방(room) 단위로 polygon 그리기
- 방 type별로 색 구분
- --target_set 인자로 어느 outputs/setN 을 읽을지 선택
```

### P6-2. 시각화 결과 검토
```
생성된 PNG를 보니까 78개 코너가 단일 방으로 묶여 있고
다중 방 구조가 안 보여. 모델이 잘못된 건지, 데이터 라벨이 잘못된 건지
어떻게 가려낼 수 있을까?
```

---

## 7. 결과 분석

### P7-1. room_type 분포 점검
```
batch 안에 들어오는 room_type 배열의 분포를 dataset 전체에 대해 집계해줘.
정상 RPLAN이면 type 1~7 이 골고루 섞여 있어야 정상.
```

### P7-2. 라벨 매핑 진단
```
type 8 (Study) 가 486회로 다른 타입(34회)에 비해 비정상적으로 많아.
100개 파일 중 66개 파일이 모든 방 type 이 8로 채워져 있는 패턴이야.
변환 스크립트의 라벨 매핑 실패가 맞는지 검증해줘.
```

### P7-3. 결론 정리
```
다음 결론을 README의 '결과 분석' 섹션으로 정리해줘:
- 1000-step diffusion 파이프라인은 정상 종료 (좌표 [-1,1] 범위 정렬됨)
- M1/MPS 포팅 자체는 성공
- 다만 데이터 라벨링 미완성으로 인해 모델이 학습 분포 밖 입력을 받아
  시각적으로 의미있는 평면도를 생성하지 못함
- 이는 데이터 도메인 차이의 문제이며, RPLAN 정식 데이터 수령 시
  코드 변경 없이 동일 파이프라인으로 재현 가능
```

---

## 8. 문서화 / 저장소 정리

### P8-1. README 작성
```
다음을 포함하는 README.md 를 작성해줘:
- 논문 / 원본 코드 출처
- 실행 환경 비교표 (원본 vs 본 재현)
- 빠른 시작 (conda env → 의존성 → 모델 배치 → 전처리 → 추론 → 시각화)
- 원본 대비 수정 사항 (파일별, 행 단위로 명시)
- 실행 로그 + 저장된 데이터 형상
- 결과 분석 (관찰 / 근본 원인 / 결론)
- 향후 계획 (RPLAN 수령 시, MagicPlan 도착 시, 캡스톤 연계)
- Acknowledgement
```

### P8-2. .gitignore 정리
```
대용량/생성물을 차단하는 .gitignore 를 추가해줘:
*.pt, *.npz, *.pkl, processed_rplan/, outputs/, ckpts/,
__pycache__/, *.pyc, *.bak*, .DS_Store, .conda/, venv/, env/
```

### P8-3. 이모티콘 제거
```
커밋 메시지와 README 본문에서 불필요한 이모티콘을 정리해줘.
(체크/엑스 같은 결과 표기는 유지)
```

### P8-4. 최종 커밋
```
지금까지 작업을 다음 순서로 커밋해줘:
1) Initial commit
2) Add house_diffusion and guided-diffusion as submodules
3) Document Apple Silicon reproduction (README)
4) Vendor house_diffusion source (서브모듈 → 일반 디렉토리 vendoring)
```

---

## 9. 업로드

### P9-1. 보고서 / 프롬프트 로그 / 코드 / 실행 결과 업로드
```
구현이 끝났으니까 GitHub 레포에 보고서, 프롬프트 로그, 구현 코드,
실행 결과를 전부 올려줘. 프롬프트 로그는 파일명을 PROMPTS.md 로.
```

---

## 참고: 사용한 도구 / 명령

- `conda create / activate` — 환경 분리
- `pip install -r requirements.txt` — 의존성
- `git submodule add` → 이후 vendoring 으로 전환
- `python -c "from house_diffusion.rplanhg_datasets import ..."` — 통계 캐시 사전 생성
- `python scripts/image_sample.py ...` — 1000-step 추론
- `python scripts/visualize_samples.py --target_set 8` — npz → PNG
- `open outputs/set8/png/` — 결과 확인

## 참고: 사용한 외부 자료

- 원본 논문: Shabani et al., *HouseDiffusion*, CVPR 2023, [arXiv:2211.13287](https://arxiv.org/abs/2211.13287)
- 원본 코드: https://github.com/aminshabani/house_diffusion
- PyTorch MPS 백엔드 가이드: https://pytorch.org/docs/stable/notes/mps.html
- guided-diffusion: https://github.com/openai/guided-diffusion
