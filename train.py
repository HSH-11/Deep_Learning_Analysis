import os
import torch
import torch.nn as nn
import torch.optim as optim
from model import PlantDiseaseCNN
from dataset import get_dataloaders


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
def train_model(model, train_loader, val_loader, criterion, optimizer, device, num_epochs=10):
    """
    정의된 학습 및 검증 루프를 통해 에포크별로 학습을 수행하고 결과를 출력하는 함수.
    """
    print(f"\nTraining started on device: {device}")
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # 1. 훈련(Train) 단계
        model.train()
        running_loss = 0.0
        total_batches = len(train_loader)
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # 그래디언트 초기화 및 역전파 학습
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
            # 배치 학습 진행률 모니터링 로그 (매 100배치마다 출력)
            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == total_batches:
                print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] | Batch [{batch_idx+1:04d}/{total_batches:04d}] | Loss: {loss.item():.4f}")
                
        epoch_train_loss = running_loss / len(train_loader.dataset)
        
        # 2. 검증(Validation) 단계
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        # 가중치 업데이트 비활성화(역전파 방지)
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct += torch.sum(preds == labels.data).item()
                total += labels.size(0)
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = correct / total
        
        # 에포크 최종 결과 소수점 4자리 포맷팅 출력
        print("=" * 65)
        print(f"Epoch [{epoch+1:02d}/{num_epochs:02d}] Results:")
        print(f"Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
        print("=" * 65 + "\n")
        
        # 가장 높은 검증 정확도를 보인 모델 가중치 저장
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"--> Best model saved with Val Acc: {best_val_acc:.4f}\n")
            
    # 학습 완료 후 최종 모델 저장
    torch.save(model.state_dict(), 'final_model.pth')
    print("Training finished. Final model saved as 'final_model.pth'")

def main():
    # 디바이스 설정 (GPU(CUDA) 또는 CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device set to: {device}")
    if torch.cuda.is_available():
        print(f"Graphics Card: {torch.cuda.get_device_name(0)}")
        
    # 데이터셋 절대경로 매핑
    base_dir = os.path.join(PROJECT_DIR, "New Plant Diseases Dataset(Augmented)", "New Plant Diseases Dataset(Augmented)")
    
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Dataset directory not found at: {base_dir}")
        
    # 1. 데이터 로더 호출 (배치 사이즈 32, num_workers=0)
    print("Loading datasets and creating DataLoader...")
    train_loader, val_loader = get_dataloaders(base_dir, batch_size=32, num_workers=0)
    
    # 2. CNN 모델 인스턴스 생성 및 디바이스 할당
    print("Initializing PlantDiseaseCNN Model...")
    model = PlantDiseaseCNN(num_classes=38).to(device)
    
    # 3. 최적화 및 오차 함수 정의 (Adam, CrossEntropyLoss)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. 학습 및 검증 루프 작동
    print("Starting optimization process...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=10
    )

if __name__ == "__main__":
    main()
