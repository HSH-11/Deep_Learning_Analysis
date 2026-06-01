# 🌿 Plant Disease Classification: 경량화 아키텍처 비교 및 딥러닝 최적화 연구

본 프로젝트는 농작물 잎 이미지를 분석하여 38가지 병해충 상태를 진단하는 딥러닝 프로젝트입니다. 
단순한 모델 적용을 넘어, **전통적인 CNN의 한계점(파라미터 폭발)을 분석하고 GAP(Global Average Pooling)를 직접 적용한 자체 최적화 모델부터 SOTA(State-of-the-Art) 모델까지 파라미터 효율성과 정확도를 비교 분석**하는 것을 핵심 연구 목표로 합니다.

---

## 📊 프로젝트 개요 및 연구 스토리라인

### Step 1. 문제 발견 (전통적 CNN의 한계)
*   **Baseline CNN**: 4개의 Conv Layer와 Dense Layer(`Flatten` 사용)로 구성된 자체 모델 설계.
*   **한계 분석**: `Flatten` 층으로 인해 파라미터가 약 2,610만 개로 폭발적으로 증가. 50층짜리 깊은 모델인 ResNet50(약 2,350만 개)보다 무거워 실무 배포(모바일/엣지 디바이스)에 부적합함을 발견.

### Step 2. 자체 아키텍처 개선 (파라미터 98% 감소 달성)
*   **Baseline CNN (GAP)**: 파라미터 폭발의 원인인 `Flatten` 대신 최신 논문 기법인 **GAP(Global Average Pooling)**를 분류기 직전에 직접 설계하여 도입.
*   **최적화 성과**: 기존 대비 파라미터를 2,610만 개 ➡️ **54만 개(약 0.5M)로 98% 대폭락** 시키며, 초경량 자체 모델 구축에 성공.

### Step 3. 최적의 실무 솔루션 검증 (SOTA 모델 적용)
*   **EfficientNet-B0 도입**: 자체 경량화 모델 연구를 바탕으로, 초기 설계부터 GAP와 Compound Scaling이 적용된 최적화 모델인 EfficientNet 도입.
*   **최종 성과**: 파라미터는 400만 개 수준으로 유지하면서 **99.73%**라는 압도적인 최종 정확도를 달성하며 '효율성'과 '성능' 두 마리 토끼를 모두 잡음. (현재 일부 모델은 학습 진행 중)

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

## 📈 주요 학습 결과 (EfficientNet-B0 기준)

- **에포크(Epochs)**: 10
- **옵티마이저 (Optimizer)**: Adam (Learning Rate: 0.001)
- **성능 기록**:
  - **Best Val Accuracy**: **99.73%** (Epoch 9)
  - **Best Val Loss**: 0.0094
  
*(※ 다른 모델들의 학습 결과는 순차적으로 업데이트 될 예정입니다.)*

---

## 📂 데이터셋 (Dataset)

- **데이터셋명**: New Plant Diseases Dataset (Augmented)
- **데이터 크기**: Train 140,590장 / Validation 35,144장 (총 38 classes)
- **데이터 증강 (Augmentation)**: Albumentations를 이용한 `Resize`, `Rotate`, `HorizontalFlip`, `ColorJitter`, `Normalize` 동적 적용.

---

## 🚀 디렉토리 구조 및 사용법

```
Deep_Learning_Analysis/
│
├── model.py                            # 5가지 아키텍처 클래스 정의 및 파라미터 검증 스크립트
├── dataset.py                          # Albumentations 파이프라인 및 DataLoader 구축
├── run_training_Baseline.ipynb         # [학습] 기존 Baseline 모델 (26M 파라미터)
├── run_training_Baseline_GAP.ipynb     # [학습] 최적화된 초경량 GAP 모델 (540K 파라미터)
├── run_training_EfficientNetB0.ipynb   # [학습] EfficientNet-B0 모델 (완료 - 99.73% 달성)
├── run_training_ResNet50.ipynb         # [학습] ResNet50 모델
├── run_training_MobileNetV3.ipynb      # [학습] MobileNetV3 모델
│
├── team_presentation.html              # 프로젝트 기획 및 결과 요약 PPT 웹 버전
└── README.md                           # 현재 설명 파일
```

*(참고: 용량이 큰 데이터셋 이미지 폴더 및 학습된 가중치 `.pth` 파일들은 `.gitignore` 규칙에 의해 저장소에 업로드되지 않습니다.)*
