# 🌿 Plant Disease Classification: 도메인 시프트 극복 및 경량화 아키텍처 최적화 연구

본 프로젝트는 농작물 잎 이미지를 분석하여 38가지 병해충 상태를 진단하는 딥러닝 프로젝트입니다. 
단순한 실험실 환경(PlantVillage)에서의 높은 정확도 달성에 그치지 않고, **실제 야외 환경(PlantDoc) 도입 시 발생하는 치명적인 도메인 시프트(Domain Shift) 문제를 Grad-CAM으로 분석하고, 이를 극복하기 위한 다양한 가설 검증 실험**을 진행한 실무 지향적 연구입니다.

---

## 📊 프로젝트 개요 및 연구 스토리라인

### Step 1. 문제 발견 (전통적 CNN의 한계)
*   **Baseline CNN**: 4개의 Conv Layer와 Dense Layer(`Flatten` 사용)로 구성된 자체 모델 설계.
*   **한계 분석**: `Flatten` 층으로 인해 파라미터가 약 2,610만 개로 폭발적으로 증가. 50층짜리 깊은 모델인 ResNet50(약 2,350만 개)보다 무거워 실무 배포(모바일/엣지 디바이스)에 부적합함을 발견.

### Step 2. 자체 아키텍처 개선 (파라미터 98% 감소 달성)
*   **Baseline CNN (GAP)**: 파라미터 폭발의 원인인 `Flatten` 대신 최신 논문 기법인 **GAP(Global Average Pooling)**를 분류기 직전에 직접 설계하여 도입.
*   **최적화 성과**: 기존 대비 파라미터를 2,610만 개 ➡️ **54만 개(약 0.5M)로 98% 대폭락** 시키며, 초경량 자체 모델 구축에 성공.

### Step 3. 다양한 모델 아키텍처 비교 실험
*   **EfficientNet-B0**: 초기 설계부터 GAP와 Compound Scaling이 적용된 최적화 모델 도입. **실험실 데이터 검증 정확도 99.73%** 달성.
*   **ResNet50**: Residual Connection 기반의 50층 심층 네트워크. Skip Connection을 통한 안정적 학습.
*   **MobileNetV3**: Depthwise Separable Convolution 기반의 경량 모델. 모바일/엣지 배포에 최적화.

### Step 4. 도메인 시프트(Domain Shift) 발생 및 XAI 원인 분석
*   **문제 발생**: 99.73%의 모델을 배경이 복잡한 **실제 야외 데이터(PlantDoc)에 테스트한 결과, 정확도가 19.12%로 폭락**.
*   **Grad-CAM 분석**: AI의 시선을 역추적한 결과, 모델이 잎의 병변을 보지 않고 **'뒷배경의 흙과 나뭇가지'에 집중하여 오답**을 내는 치명적 과적합 현상(환경 문맥 의존) 발견.

### Step 5. 도메인 시프트 극복을 위한 가설 검증 실험
문제를 해결하기 위해 3가지 가설을 세우고 실험을 진행했습니다.
*   **🧪 가설 1 (물리적 전처리 / V3)**: AI로 배경(rembg)을 지우면 해결될 것이다.
    *   **결과**: 47.81% (실패). 잎 가장자리의 병변이 훼손되고 상황적 문맥이 삭제되어 한계 노출.
*   **🧪 가설 2 (혼합 도메인 학습 / V4)**: 실험실 데이터와 야외 데이터를 섞어 학습하면 해결될 것이다.
    *   **결과**: 47.01% (실패). 쉬운 데이터가 섞이면서 야외 환경 적응에 대한 모델의 시선이 분산됨.
*   **🏆 가설 3 (본질적 도메인 적응 / V2)**: **원본 야외 데이터만 투입하되 가혹한 증강(ColorJitter 등)과 수학적 최적화(Cosine Annealing, Label Smoothing)로 정면 승부한다.**
    *   **결과**: **60.56% (성공)**. 꼼수(배경 제거, 혼합)를 배제하고 모델 스스로 가혹한 환경을 이겨내도록 유도한 정공법이 3배 이상의 성능 향상을 이끌어 냄.

### Step 6. Vision Transformer(ViT) 아키텍처 비교
*   **동일 프로토콜 비교**: PlantDoc train/test, 동일 증강·Label Smoothing·Cosine Annealing·2단계 학습(헤드 5ep → 전체 25ep) 조건에서 EfficientNet-B0와 ViT-Small을 공정 비교.
*   **성능**: EfficientNet-B0 **65.34%** (164/251) vs ViT-Small **70.92%** (178/251), **+5.58%p** 우위.
*   **이미지별 비교**: 251장 중 둘 다 정답 149 · ViT만 정답 29 · EfficientNet만 정답 15.
*   **ViT Attention 시각화**: CLS 토큰 attention map으로 파인튜닝 후 병변 집중이 Grad-CAM 대비 개선됨을 확인. 일부 오답에서는 배경 shortcut·유사 병해 혼동 잔존.

---

## 🏗️ 모델 아키텍처 및 파라미터 비교

| 모델명 | 분류기 구조 특징 | 파라미터 수 (개) | 실험실 성능 | 평가 |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline CNN (기존)** | `Flatten` -> Dense | 26,099,494 | ✅ 학습 완료 | 모델의 깊이에 비해 연산량이 가장 무거움 |
| **ResNet50** | GAP 적용 + Residual | 23,585,894 | ✅ 학습 완료 | Skip Connection으로 깊은 학습이 안정적 |
| **MobileNetV3** | GAP + Depthwise Conv | 4,250,710 | ✅ 학습 완료 | 가볍고 모바일 배포에 최적화 |
| **EfficientNet-B0** | **GAP + Compound Scaling** | **4,056,226** | ✅ **99.73%** | 가벼운 파라미터 + **압도적 성능** |
| **ViT-Small** | Self-Attention + Patch Embedding | 21,676,642 | ✅ **70.92%** (PlantDoc) | 야외 도메인 적응 **현재 최고** |
| **Baseline CNN (GAP 적용)** | **`GAP` -> Dense** | **540,454** | ✅ 학습 완료 | 파라미터 구조 개선으로 인한 **극강의 초경량화** |

---

## 📂 데이터셋 (Dataset)

*   **학습/검증 (Source Domain)**: New Plant Diseases Dataset (Augmented) - 실험실 단색 배경 (175,734장)
*   **테스트 (Target Domain)**: PlantDoc Dataset - 실제 야외 밭 환경
*   **데이터 증강 (Augmentation)**: Albumentations를 이용한 `Resize`, `Rotate`, `HorizontalFlip`, `ColorJitter`, `Normalize` 동적 적용.

---

## 🚀 디렉토리 구조

```
Deep_Learning_Analysis/
│
├── 🧠 핵심 모듈 (Core Modules)
│   ├── model.py                          # 6가지 모델 아키텍처 클래스 정의 (ViT-Small 포함)
│   ├── dataset.py                        # Albumentations 전처리 파이프라인 및 DataLoader 구축
│   ├── train.py                          # 기본 터미널 학습 스크립트
│   └── evaluate_plantdoc.py              # 야외 데이터(PlantDoc) 성능 평가
│
├── 📓 모델별 학습 노트북 (Training Notebooks)
│   ├── run_training_Baseline.ipynb       # Baseline CNN 학습 (26M 파라미터)
│   ├── run_training_Baseline_GAP.ipynb   # 초경량 GAP 모델 학습 (540K 파라미터)
│   ├── run_training_EfficientNetB0.ipynb # EfficientNet-B0 학습 (99.73%)
│   ├── run_training_ResNet50.ipynb       # ResNet50 학습
│   └── run_training_MobileNetV3.ipynb    # MobileNetV3 학습
│
├── 🧪 도메인 시프트 극복 실험 (Fine-tuning)
│   ├── finetune_plantdoc.py / v1.ipynb   # [V1] 단순 전이학습 시도
│   ├── finetune_plantdoc_v2.py / .ipynb  # [V2] 가설 3 ✅ 강한 증강 + 수학적 최적화 (60.56%)
│   ├── finetune_plantdoc_v3.py / .ipynb  # [V3] 가설 1 ❌ rembg 배경 물리적 제거
│   ├── finetune_plantdoc_v4.py / .ipynb  # [V4] 가설 2 ❌ 실험실+야외 혼합 데이터 학습
│   ├── finetune_plantdoc_vit.py          # ViT-Small PlantDoc 파인튜닝 (70.92%)
│   ├── finetune_plantdoc_efficientnet_match_vit.py  # ViT와 동일 프로토콜 EfficientNet 재학습 (65.34%)
│   ├── evaluate_plantdoc_vit.py        # ViT-Small PlantDoc test 평가
│   ├── generate_vit_attention.py       # ViT CLS attention 시각화
│   ├── preprocess_rembg_all.py           # V3 실험용 배경 제거 전처리 스크립트
│   └── generate_gradcam.py               # Grad-CAM XAI 시각화 분석 스크립트
│
├── 📊 결과 및 시각화 에셋
│   ├── gradcam_results/                  # Grad-CAM 시각화 오답 분석 결과 이미지
│   ├── vit_attention_results/            # ViT Attention 시각화 (정답 5 + 오답 5)
│   ├── training_results*.png             # 모델별 학습 곡선 그래프
│   └── assets/                           # 학습 곡선(V2~V4) 및 PPT용 에셋
│
├── 🎤 발표 자료
│   └── team_presentation.html            # 💡 [웹 발표용] 전체 연구 서사 PPT
│
└── 🧰 기타
    ├── dataset_dataloader_test.ipynb      # 데이터셋 로딩 테스트 및 시각화
    └── .gitignore                         # 대용량 파일 제외 규칙
```

---

## 📝 파일 상세 설명

### 🧠 핵심 모듈

| 파일 | 설명 |
|---|---|
| **`model.py`** | 6가지 모델 아키텍처(`PlantDiseaseCNN`, `PlantDiseaseCNN_GAP`, `PlantDiseaseResNet50`, `PlantDiseaseEfficientNetB0`, `PlantDiseaseViTSmall`, `PlantDiseaseMobileNetV3`)를 `nn.Module` 클래스로 정의. ViT-Small은 `timm` 기반 `vit_small_patch16_224` 사용. |
| **`dataset.py`** | Albumentations 라이브러리를 활용한 이미지 전처리(Resize 224×224, Normalize, 증강) 파이프라인 및 PyTorch `DataLoader`를 구축하는 함수(`get_dataloaders`) 제공. Train/Validation 폴더를 자동 분리하여 로드. |
| **`train.py`** | 터미널에서 직접 실행할 수 있는 학습 스크립트. 에포크별 Train Loss, Val Loss, Val Accuracy를 출력하고 Best 모델을 자동 저장하는 범용 학습 루틴. |
| **`evaluate_plantdoc.py`** | 학습된 모델(`.pth`)을 불러와 야외 데이터(PlantDoc)에서의 Top-1 정확도를 측정하는 평가 스크립트. 도메인 시프트 정도를 수치적으로 확인하는 데 사용. |

### 📓 학습 노트북

| 파일 | 설명 |
|---|---|
| **`run_training_Baseline.ipynb`** | 4-Block Conv + Flatten + Dense로 구성된 기존 Baseline CNN 학습. 파라미터 26M개의 무거운 구조. 학습 그래프(Loss/Accuracy) 포함. |
| **`run_training_Baseline_GAP.ipynb`** | Flatten을 GAP(Global Average Pooling)로 대체한 경량화 모델 학습. 파라미터를 540K개로 **98% 감소** 달성. |
| **`run_training_EfficientNetB0.ipynb`** | Compound Scaling 기반 EfficientNet-B0 전이학습. 4M 파라미터로 **실험실 Val Acc 99.73%** 달성. |
| **`run_training_ResNet50.ipynb`** | 50층 Residual Network 전이학습. Skip Connection을 통한 안정적인 깊은 학습 수행. |
| **`run_training_MobileNetV3.ipynb`** | Depthwise Separable Convolution 기반 경량 모델 전이학습. 모바일/엣지 디바이스 배포 시나리오 검증. |

### 🧪 도메인 시프트 극복 실험

| 파일 | 설명 |
|---|---|
| **`finetune_plantdoc.py` / `v1.ipynb`** | 실험실에서 학습된 EfficientNet-B0을 야외 데이터로 단순 파인튜닝하는 기본 실험. |
| **`finetune_plantdoc_v2.py` / `.ipynb`** | **🏆 최종 성공 실험.** 강한 이미지 증강(ColorJitter, RandomRotation) + Cosine Annealing LR Scheduler + Label Smoothing을 적용하여 **야외 정확도 60.56%** 달성 (3배 이상 향상). |
| **`finetune_plantdoc_v3.py` / `.ipynb`** | rembg 라이브러리로 배경을 물리적으로 제거한 데이터로 학습. 잎 경계 훼손으로 인해 47.81%에 그침 (실패). |
| **`finetune_plantdoc_v4.py` / `.ipynb`** | 실험실 데이터(PlantVillage)와 야외 데이터(PlantDoc)를 혼합하여 학습. 쉬운 데이터의 간섭으로 47.01% (실패). |
| **`preprocess_rembg_all.py`** | V3 실험을 위해 PlantDoc 데이터셋 전체 이미지의 배경을 rembg로 자동 제거하는 배치 전처리 스크립트. |
| **`generate_gradcam.py`** | 학습된 모델의 마지막 Conv 층에서 Grad-CAM 히트맵을 추출하여, 모델이 이미지의 어느 부분을 보고 판단했는지를 시각화. 오답 원인 분석에 사용. |
| **`finetune_plantdoc_vit.py`** | ViT-Small PlantDoc 도메인 적응. 2단계 파인튜닝(헤드 freeze → 전체 unfreeze) + Label Smoothing + Cosine Annealing. **PlantDoc test 70.92%**. |
| **`finetune_plantdoc_efficientnet_match_vit.py`** | ViT 실험과 동일 프로토콜로 EfficientNet-B0 재학습. 공정 비교 기준선 **65.34%**. |
| **`evaluate_plantdoc_vit.py`** | `finetuned_model_vit_small_v2.pth` 체크포인트로 PlantDoc test 정확도 평가. |
| **`generate_vit_attention.py`** | 마지막 ViT 블록 CLS→패치 attention map 추출. `vit_attention_results/`에 정답·오답 샘플 저장. |

### 📊 결과물

| 파일/폴더 | 설명 |
|---|---|
| **`training_results*.png`** | 각 모델의 학습 과정(Train/Val Loss, Val Accuracy)을 시각화한 그래프 이미지. |
| **`gradcam_results/`** | Grad-CAM 분석 결과 이미지 5장. 모델이 잎의 병변 대신 배경에 집중하는 도메인 시프트 현상을 시각적으로 증명. |
| **`vit_attention_results/`** | ViT Attention 시각화 10장(정답 5, 오답 5). 파인튜닝 후 병변 집중·배경 shortcut 패턴 분석. |
| **`assets/`** | 파인튜닝 실험(V2~V4)의 학습 곡선 그래프 및 발표용 에셋. |

---

## ⚙️ 실행 환경

*   **Python**: 3.10+
*   **Framework**: PyTorch
*   **전처리**: Albumentations
*   **시각화**: Matplotlib, Grad-CAM
*   **OS**: Windows (DataLoader `num_workers=0` 설정)

---

*(참고: 용량이 큰 데이터셋은 `.gitignore` 규칙에 의해 저장소에 업로드되지 않습니다. 가중치 `.pth` 파일도 제외됩니다.)*
