"""
ViT-Small + V2 Pipeline for PlantDoc Domain Adaptation
======================================================
EfficientNet V2(60.56%) 대비 ViT-Small의 야외 적응 성능을 검증합니다.

전략 (추천 실험 1단계):
1. V2와 동일한 가혹한 증강 + Label Smoothing + Cosine Annealing
2. 소량 데이터에 맞춘 2-Phase Fine-tuning
   - Phase 1: head만 학습 (backbone freeze)
   - Phase 2: backbone + head differential LR fine-tune
"""
import os
import glob

import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from model import PlantDiseaseViTSmall

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ORIGINAL_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]
class_to_idx = {cls_name: i for i, cls_name in enumerate(ORIGINAL_CLASSES)}

CHECKPOINT_PATH = os.path.join(PROJECT_DIR, "finetuned_model_vit_small_v2.pth")
PHASE1_EPOCHS = 5
PHASE2_EPOCHS = 25
BATCH_SIZE = 16 if device.type == "cuda" else 8


class PlantDocMappedDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path) and folder_name in class_to_idx:
                target_idx = class_to_idx[folder_name]
                for img_path in glob.glob(os.path.join(folder_path, "*.*")):
                    if img_path.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.image_paths.append(img_path)
                        self.labels.append(target_idx)

        print(f"Loaded {len(self.image_paths)} images from {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


train_transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total if total else 0.0


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Train", leave=False)
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Acc": f"{100 * correct / total:.2f}%"})

    return running_loss / len(loader), 100 * correct / total


def build_phase2_optimizer(model):
    head_params = []
    backbone_params = []
    for name, param in model.model.named_parameters():
        if name.startswith("head"):
            head_params.append(param)
        else:
            backbone_params.append(param)
    return optim.Adam(
        [
            {"params": backbone_params, "lr": 5e-5},
            {"params": head_params, "lr": 3e-4},
        ]
    )


def main():
    train_dir = os.path.join(PROJECT_DIR, "plantdoc", "train")
    test_dir = os.path.join(PROJECT_DIR, "plantdoc", "test")

    train_loader = DataLoader(
        PlantDocMappedDataset(train_dir, transform=train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        PlantDocMappedDataset(test_dir, transform=test_transform),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = PlantDiseaseViTSmall(num_classes=38, pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    best_acc = 0.0

    print(f"\n=== Phase 1: Head-only fine-tuning ({PHASE1_EPOCHS} epochs) ===")
    model.freeze_backbone()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    for epoch in range(PHASE1_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_acc = evaluate(model, test_loader)
        print(
            f"Phase1 Epoch [{epoch + 1}/{PHASE1_EPOCHS}] "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> Best model saved: {best_acc:.2f}%")

    print(f"\n=== Phase 2: Full fine-tuning ({PHASE2_EPOCHS} epochs) ===")
    model.unfreeze_all()
    optimizer = build_phase2_optimizer(model)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS)

    for epoch in range(PHASE2_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        scheduler.step()
        val_acc = evaluate(model, test_loader)
        lr_head = optimizer.param_groups[1]["lr"]
        lr_backbone = optimizer.param_groups[0]["lr"]
        print(
            f"Phase2 Epoch [{epoch + 1}/{PHASE2_EPOCHS}] "
            f"LR(head/backbone): {lr_head:.6f}/{lr_backbone:.6f} | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> Best model saved: {best_acc:.2f}%")

    print(f"\nViT-Small fine-tuning complete. Best Validation Accuracy: {best_acc:.2f}%")
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print("Compare with EfficientNet V2 baseline: 60.56%")


if __name__ == "__main__":
    main()
