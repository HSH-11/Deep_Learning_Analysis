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

### Step 3. 실험실 모델 구축 완료 (EfficientNet-B0 도입)
*   **SOTA 모델 검증**: 자체 경량화 모델 연구를 바탕으로, 초기 설계부터 GAP와 Compound Scaling이 적용된 최적화 모델인 EfficientNet-B0 도입.
*   **1차 성과**: 파라미터 400만 개 수준으로 가벼움을 유지하며 **실험실 데이터(PlantVillage) 검증 정확도 99.73%** 라는 압도적 성능 달성.

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

---

## 🏗️ 모델 아키텍처 및 파라미터 비교

| 모델명 | 분류기 구조 특징 | 파라미터 수 (개) | 평가 |
| :--- | :--- | :--- | :--- |
| **Baseline CNN (기존)** | `Flatten` -> Dense | 26,099,494 | 모델의 깊이에 비해 연산량이 가장 무거움 |
| ResNet50 (진행 예정) | GAP 적용 + Residual | 23,585,894 | 깊은 층으로 인해 무거운 편 |
| MobileNetV3 (진행 예정) | GAP + Depthwise Conv | 4,250,710 | 가볍고 효율적임 |
| **EfficientNet-B0 (완료)** | **GAP + Compound Scaling** | **4,056,226** | 가벼운 파라미터 + **압도적 성능(99.73%)** |
| **Baseline CNN (GAP 적용)** | **`GAP` -> Dense** | **540,454** | 파라미터 구조 개선으로 인한 **극강의 초경량화** |

---

## 📂 데이터셋 (Dataset)

*   **학습/검증 (Source Domain)**: New Plant Diseases Dataset (Augmented) - 실험실 단색 배경 (175,734장)
*   **테스트 (Target Domain)**: PlantDoc Dataset - 실제 야외 밭 환경
*   **데이터 증강 (Augmentation)**: Albumentations를 이용한 `Resize`, `Rotate`, `HorizontalFlip`, `ColorJitter`, `Normalize` 동적 적용.

---

## 🚀 주요 파일 및 디렉토리 구조

```
Deep_Learning_Analysis/
│
├── 🧠 핵심 모듈
│   ├── model.py                          # 5가지 아키텍처 클래스 정의
│   ├── dataset.py                        # Albumentations 파이프라인 및 DataLoader 구축
│   ├── train.py                          # 기본 터미널 학습 스크립트
│   └── evaluate_plantdoc.py              # 야외 데이터(PlantDoc) 검증 및 성능 측정
│
├── 📓 학습 노트북 (Baseline)
│   ├── run_training_Baseline.ipynb       # 기존 Baseline 모델 학습 (26M 파라미터)
│   ├── run_training_Baseline_GAP.ipynb   # 최적화된 초경량 GAP 모델 (540K 파라미터)
│   └── run_training_EfficientNetB0.ipynb # EfficientNet-B0 모델 (실험실 99.73%)
│
├── 🧪 도메인 시프트 극복 실험 (Fine-tuning)
│   ├── finetune_plantdoc.py / .ipynb     # [V1] 단순 전이학습 시도
│   ├── finetune_plantdoc_v2.py / .ipynb  # [V2] 가설 3 성공: 강한 증강 + 수학적 최적화 (60.56%)
│   ├── finetune_plantdoc_v3.py / .ipynb  # [V3] 가설 1 실패: rembg 배경 물리적 제거
│   ├── finetune_plantdoc_v4.py / .ipynb  # [V4] 가설 2 실패: 실험실+야외 혼합 데이터 학습
│   ├── preprocess_rembg_all.py           # V3 실험용 배경 제거 전처리 스크립트
│   └── generate_gradcam.py               # Grad-CAM XAI 시각화 분석 스크립트
│
├── 📊 결과 및 시각화 에셋
│   ├── gradcam_results/                  # Grad-CAM 시각화 오답 분석 결과 이미지
│   └── assets/                           # 학습 곡선(V2~V4) 및 PPT용 에셋
│
└── 🎤 발표 자료
    ├── team_presentation.html            # 💡 [웹 발표용] 화려한 디자인의 전체 연구 서사 PPT
    ├── print_team_presentation.html      # 🖨️ [인쇄용] PDF 저장을 위한 세로 스크롤 버전
    └── Final_Presentation_Updated.pptx   # 📝 [편집용] 기존 PPTX 템플릿에 신규 서사 삽입본
```

*(참고: 용량이 큰 데이터셋 `.gitignore` 규칙에 의해 저장소에 업로드되지 않습니다. 가중치 `.pth` 파일도 제외됩니다.)*
