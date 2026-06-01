import os
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

VALID_SUFFIXES = {".jpg", ".jpeg", ".png"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PROJECT_ROOT = Path(__file__).resolve().parent


class PlantDiseaseDataset(Dataset):
    """
    농작물 잎 병해충 진단을 위한 커스텀 PyTorch Dataset 클래스.
    Albumentations 전처리/증강 라이브러리를 적용합니다.
    """

    def __init__(self, root_dir, transform=None, class_to_idx=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        if class_to_idx is not None:
            self.class_to_idx = dict(class_to_idx)
            self.classes = [
                name for name, _ in sorted(self.class_to_idx.items(), key=lambda item: item[1])
            ]
        else:
            self.classes = sorted(
                d
                for d in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, d))
            )
            self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}

        present_classes = []
        for cls_name in sorted(os.listdir(root_dir)):
            cls_dir = os.path.join(root_dir, cls_name)
            if not os.path.isdir(cls_dir):
                continue
            if cls_name not in self.class_to_idx:
                print(f"[Warning] Skipping unknown class folder: {cls_name}")
                continue

            present_classes.append(cls_name)
            for fname in sorted(os.listdir(cls_dir)):
                if os.path.splitext(fname)[1].lower() not in VALID_SUFFIXES:
                    continue
                img_path = os.path.join(cls_dir, fname)
                if not os.path.isfile(img_path):
                    continue
                self.image_paths.append(img_path)
                self.labels.append(self.class_to_idx[cls_name])

        print(
            f"Loaded dataset from {root_dir}: {len(self.image_paths)} images, "
            f"{len(present_classes)} / {len(self.classes)} classes present."
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {img_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


def get_class_to_idx_from_dir(train_dir: str) -> dict[str, int]:
    """NPD train 폴더 기준 38클래스 라벨 매핑 (sorted 폴더명 순)."""
    classes = sorted(
        d
        for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    )
    return {cls_name: i for i, cls_name in enumerate(classes)}


def _resolve_npd_train_dir(reference_train_dir: str | None) -> str:
    if reference_train_dir is not None:
        if not os.path.isdir(reference_train_dir):
            raise FileNotFoundError(f"reference_train_dir not found: {reference_train_dir}")
        return reference_train_dir

    candidates = [
        PROJECT_ROOT
        / "New Plant Diseases Dataset(Augmented)"
        / "New Plant Diseases Dataset(Augmented)"
        / "train",
        Path(
            r"c:\Users\user\Deep_Learning_Analysis\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
        ),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return str(candidate)

    raise FileNotFoundError(
        "NPD train directory not found. Pass reference_train_dir explicitly."
    )


def get_transforms(augment_level: str = "full", image_size: int = 224):
    """
    Albumentations 전처리/증강 파이프라인.

    augment_level:
        - "full": NPD 기본 (Rotate, Flip, ColorJitter)
        - "minimal": Resize + Normalize only
        - "plantdoc": 현장 데이터 fine-tune용 (약한 Flip/Rotate, ColorJitter 없음)
    """
    normalize = A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    val_transform = A.Compose(
        [
            A.Resize(image_size, image_size),
            normalize,
            ToTensorV2(),
        ]
    )

    if augment_level == "minimal":
        train_transform = val_transform
    elif augment_level == "plantdoc":
        train_transform = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                normalize,
                ToTensorV2(),
            ]
        )
    else:
        train_transform = A.Compose(
            [
                A.Resize(image_size, image_size),
                A.Rotate(limit=30, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5
                ),
                normalize,
                ToTensorV2(),
            ]
        )

    return train_transform, val_transform


def _make_dataloader(dataset, batch_size, shuffle, num_workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=num_workers > 0,
    )


def get_dataloaders(base_dir, batch_size=32, num_workers=0, augment_level="full", image_size=224):
    """
    New Plant Diseases Dataset (train / valid) DataLoader.

    Args:
        base_dir: 'train', 'valid' 폴더를 포함하는 상위 경로
    """
    train_dir = os.path.join(base_dir, "train")
    valid_dir = os.path.join(base_dir, "valid")

    train_transform, val_transform = get_transforms(
        augment_level=augment_level, image_size=image_size
    )

    train_dataset = PlantDiseaseDataset(train_dir, transform=train_transform)
    val_dataset = PlantDiseaseDataset(valid_dir, transform=val_transform)

    train_loader = _make_dataloader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = _make_dataloader(
        val_dataset, batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader


def get_plantdoc_dataloaders(
    plantdoc_dir,
    reference_train_dir=None,
    batch_size=32,
    num_workers=0,
    image_size=224,
):
    """
    PlantDoc DataLoader (train / test).

    PlantDoc 폴더명은 PlantVillage 38클래스 형식이어야 합니다.
    라벨 인덱스는 NPD train 폴더와 동일하게 맞춥니다 (38-class 모델 fine-tune용).

    Args:
        plantdoc_dir: 'train', 'test' 폴더를 포함하는 plantdoc 루트 경로
        reference_train_dir: NPD train 경로 (38클래스 class_to_idx 기준). None이면 자동 탐색.
        batch_size: 미니배치 크기
        num_workers: DataLoader worker 수 (Windows: 0 권장)
        image_size: Resize 대상 크기 (기본 224)

    Returns:
        train_loader, test_loader, class_to_idx
    """
    plantdoc_dir = str(plantdoc_dir)
    train_dir = os.path.join(plantdoc_dir, "train")
    test_dir = os.path.join(plantdoc_dir, "test")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"PlantDoc train directory not found: {train_dir}")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"PlantDoc test directory not found: {test_dir}")

    npd_train_dir = _resolve_npd_train_dir(reference_train_dir)
    class_to_idx = get_class_to_idx_from_dir(npd_train_dir)
    print(f"Using {len(class_to_idx)}-class label mapping from: {npd_train_dir}")

    train_transform, test_transform = get_transforms(
        augment_level="plantdoc", image_size=image_size
    )

    train_dataset = PlantDiseaseDataset(
        train_dir, transform=train_transform, class_to_idx=class_to_idx
    )
    test_dataset = PlantDiseaseDataset(
        test_dir, transform=test_transform, class_to_idx=class_to_idx
    )

    train_loader = _make_dataloader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = _make_dataloader(
        test_dataset, batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader, class_to_idx


if __name__ == "__main__":
    dataset_base_dir = (
        PROJECT_ROOT
        / "New Plant Diseases Dataset(Augmented)"
        / "New Plant Diseases Dataset(Augmented)"
    )
    plantdoc_dir = PROJECT_ROOT / "plantdoc"

    if dataset_base_dir.exists():
        print("--- NPD DataLoader test ---")
        train_loader, val_loader = get_dataloaders(
            str(dataset_base_dir), batch_size=8, num_workers=0
        )
        images, labels = next(iter(train_loader))
        print(f"NPD batch: images={images.shape}, labels={labels.shape}")

    if plantdoc_dir.exists():
        print("\n--- PlantDoc DataLoader test ---")
        train_loader, test_loader, class_to_idx = get_plantdoc_dataloaders(
            str(plantdoc_dir), batch_size=8, num_workers=0
        )
        images, labels = next(iter(train_loader))
        print(f"PlantDoc batch: images={images.shape}, labels={labels.shape}")
        print(f"num_classes={len(class_to_idx)}, label sample={labels[:5].tolist()}")
