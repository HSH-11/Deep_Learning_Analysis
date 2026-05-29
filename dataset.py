import os
import glob
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

class PlantDiseaseDataset(Dataset):
    """
    농작물 잎 병해충 진단을 위한 커스텀 PyTorch Dataset 클래스.
    Albumentations 전처리/증강 라이브러리를 적용합니다.
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 클래스 폴더명 정렬하여 일관된 라벨 매핑 생성
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 지원할 이미지 확장자 리스트
        valid_extensions = ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG')
        
        # 모든 클래스 폴더 내의 이미지 파일 경로 및 해당 라벨 매칭 수집
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            
            for ext in valid_extensions:
                pattern = os.path.join(cls_dir, ext)
                for img_path in glob.glob(pattern):
                    self.image_paths.append(img_path)
                    self.labels.append(self.class_to_idx[cls_name])
                    
        print(f"Loaded dataset from {root_dir}: {len(self.image_paths)} images found across {len(self.classes)} classes.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # OpenCV를 사용해 이미지 로드 및 BGR -> RGB 채널 변환
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Albumentations 변환 적용 (있을 경우)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label

def get_transforms():
    """
    Albumentations 이미지 전처리 및 증강(Augmentation) 정의 파이프라인.
    """
    # 1. 학습(Train)용 전처리 및 증강 파이프라인
    train_transform = A.Compose([
        A.Resize(224, 224),
        A.Rotate(limit=30, p=0.5), # RandomRotation
        A.HorizontalFlip(p=0.5), # HorizontalFlip
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5), # ColorJitter
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        ), # ImageNet 표준 정규화
        ToTensorV2() # PyTorch 텐서 변환
    ])
    
    # 2. 검증(Validation)용 전처리 파이프라인 (증강 미적용)
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=(0.485, 0.456, 0.406), 
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2()
    ])
    
    return train_transform, val_transform

def get_dataloaders(base_dir, batch_size=32, num_workers=0):
    """
    학습 및 검증용 데이터로더를 생성하여 반환하는 빌더 함수.
    
    Args:
        base_dir (str): 'train'과 'valid' 폴더를 포함하는 상위 데이터셋 디렉토리 경로
        batch_size (int): 미니배치 크기
        num_workers (int): 멀티프로세싱을 위한 워커 스레드 개수 (Windows는 0 또는 2 추천)
        
    Returns:
        train_loader (DataLoader), val_loader (DataLoader)
    """
    train_dir = os.path.join(base_dir, 'train')
    valid_dir = os.path.join(base_dir, 'valid')
    
    train_transform, val_transform = get_transforms()
    
    train_dataset = PlantDiseaseDataset(train_dir, transform=train_transform)
    val_dataset = PlantDiseaseDataset(valid_dir, transform=val_transform)
    
    # DataLoader 정의
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader

# 로컬 환경 데이터셋 연결 및 형태 검증 테스트 코드
if __name__ == "__main__":
    # 프로젝트 내 실제 데이터셋 기본 경로 설정
    dataset_base_dir = r"c:\Users\dltjr\Desktop\project\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)"
    
    if os.path.exists(dataset_base_dir):
        print("--- 데이터 파이프라인 검증 테스트 ---")
        try:
            # 배치 사이즈 8로 설정해 테스트 로딩 실행
            train_loader, val_loader = get_dataloaders(dataset_base_dir, batch_size=8, num_workers=0)
            
            # 첫 번째 배치 추출 테스트
            images, labels = next(iter(train_loader))
            
            print("\n[성공] 데이터로더 작동 검증 완료!")
            print(f"추출된 이미지 배치 형태 (Shape): {images.shape} (기대값: [8, 3, 224, 224])")
            print(f"추출된 라벨 배치 형태 (Shape): {labels.shape} (기대값: [8])")
            print(f"라벨 값 샘플: {labels.tolist()}")
        except Exception as e:
            print(f"\n[에러] 데이터로더 테스트 중 오류가 발생했습니다: {e}")
    else:
        print(f"[Warning] 테스트를 진행할 수 없습니다. 경로가 존재하지 않습니다: {dataset_base_dir}")
