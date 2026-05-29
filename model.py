import torch
import torch.nn as nn

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

# 모델의 형태(Shape) 출력 및 오류 발생 여부 검증용 테스트 코드
if __name__ == "__main__":
    # 1. 모델 인스턴스 생성 (최종 분류 클래스: 38개)
    model = PlantDiseaseCNN(num_classes=38)
    
    # 2. 임의의 더미 입력 텐서 생성 (배치 크기 8, RGB 3채널, 224x224 크기)
    dummy_input = torch.randn(8, 3, 224, 224)
    
    print("--- 모델 형태 검증 테스트 ---")
    print(f"입력 텐서 형태 (dummy_input shape): {dummy_input.shape}")
    
    # 3. 모델 순전파(forward) 연산 수행
    try:
        output = model(dummy_input)
        print(f"출력 텐서 형태 (output shape):       {output.shape} (기대값: [8, 38])")
        print("\n검증 성공! 에러 없이 정상적으로 연산이 완료되었습니다.")
    except Exception as e:
        print(f"\n검증 실패! 에러가 발생했습니다: {e}")
