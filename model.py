import torch
import torch.nn as nn
import torchvision.models as models

class PlantDiseaseCNN(nn.Module):
    def __init__(self, num_classes=38):
        super(PlantDiseaseCNN, self).__init__()
        
        # Conv Block 1: Input (3, 224, 224) -> Output (32, 112, 112)
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Conv Block 2: Input (32, 112, 112) -> Output (64, 56, 56)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Conv Block 3: Input (64, 56, 56) -> Output (128, 28, 28)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Conv Block 4: Input (128, 28, 28) -> Output (256, 14, 14)
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classifier: Linear 분류기 구성
        # 4번의 Pooling을 거쳐 크기가 224 -> 112 -> 56 -> 28 -> 14로 줄어듭니다.
        # 최종 Conv Block의 출력 형태는 [Batch, 256, 14, 14]이므로 Flatten 시 256 * 14 * 14 차원이 됩니다.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.classifier(x)
        return x

# ==========================================
# 기존 CNN에 GAP(Global Average Pooling) 적용 버전
# ==========================================
class PlantDiseaseCNN_GAP(nn.Module):
    def __init__(self, num_classes=38):
        super(PlantDiseaseCNN_GAP, self).__init__()
        
        # Conv Block들은 기존과 동일
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Classifier: Flatten 대신 GAP 적용
        # GAP를 거치면 [Batch, 256, 14, 14] -> [Batch, 256, 1, 1]이 됨
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), # 이게 GAP 역할입니다
            nn.Flatten(),                 # 1x1을 1차원으로 변환
            nn.Linear(256, 512),          # 256 * 14 * 14 (50176)에서 256으로 극단적 축소!
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        x = self.classifier(x)
        return x

# ==========================================
# 추가 모델 1: ResNet50 (Transfer Learning)
# ==========================================
class PlantDiseaseResNet50(nn.Module):
    def __init__(self, num_classes=38):
        super(PlantDiseaseResNet50, self).__init__()
        # ImageNet 사전학습 가중치 로드
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # 마지막 분류기(fc) 교체
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)

# ==========================================
# 추가 모델 2: EfficientNet-B0 (Transfer Learning)
# ==========================================
class PlantDiseaseEfficientNetB0(nn.Module):
    def __init__(self, num_classes=38):
        super(PlantDiseaseEfficientNetB0, self).__init__()
        # ImageNet 사전학습 가중치 로드
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # 마지막 분류기(classifier) 교체
        num_ftrs = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)

# ==========================================
# 추가 모델 3: MobileNetV3-Large (Transfer Learning)
# ==========================================
class PlantDiseaseMobileNetV3(nn.Module):
    def __init__(self, num_classes=38):
        super(PlantDiseaseMobileNetV3, self).__init__()
        # ImageNet 사전학습 가중치 로드
        self.model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        
        # 마지막 분류기(classifier) 교체
        num_ftrs = self.model.classifier[3].in_features
        self.model.classifier[3] = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)


# 모델의 형태(Shape) 출력 및 오류 발생 여부 검증용 테스트 코드
if __name__ == "__main__":
    # 테스트할 모델 리스트
    model_dict = {
        "Baseline CNN": PlantDiseaseCNN(num_classes=38),
        "Baseline CNN (GAP)": PlantDiseaseCNN_GAP(num_classes=38),
        "ResNet50": PlantDiseaseResNet50(num_classes=38),
        "EfficientNet-B0": PlantDiseaseEfficientNetB0(num_classes=38),
        "MobileNetV3": PlantDiseaseMobileNetV3(num_classes=38)
    }
    
    # 임의의 더미 입력 텐서 생성 (배치 크기 8, RGB 3채널, 224x224 크기)
    dummy_input = torch.randn(8, 3, 224, 224)
    
    print("--- 모델 형태 검증 및 파라미터 수 확인 ---")
    print(f"입력 텐서 형태 (dummy_input shape): {dummy_input.shape}\n")
    
    for name, model in model_dict.items():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[{name}] 파라미터 수: {total_params:,} 개")
        try:
            output = model(dummy_input)
            if output.shape == (8, 38):
                print("  -> 검증 성공! (출력 형태 [8, 38] 확인됨)\n")
            else:
                print(f"  -> 검증 실패 (형태 불일치: {output.shape})!\n")
        except Exception as e:
            print(f"  -> 검증 실패! 에러 발생: {e}\n")
