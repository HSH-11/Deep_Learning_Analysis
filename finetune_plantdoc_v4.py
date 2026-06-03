import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from PIL import Image
import glob
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import PlantDiseaseEfficientNetB0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

ORIGINAL_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot', 
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]
class_to_idx = {cls_name: i for i, cls_name in enumerate(ORIGINAL_CLASSES)}

class PlantDocMappedDataset(Dataset):
    """PlantDoc (outdoor) dataset - maps folder names to original 38-class indices."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                if folder_name in class_to_idx:
                    target_idx = class_to_idx[folder_name]
                    for img_path in glob.glob(os.path.join(folder_path, "*.*")):
                        if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append(img_path)
                            self.labels.append(target_idx)
                            
        print(f"[PlantDoc] Loaded {len(self.image_paths)} images from {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

class LabDataset(Dataset):
    """Lab (PlantVillage) dataset - maps folder names to original 38-class indices."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for folder_name in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                if folder_name in class_to_idx:
                    target_idx = class_to_idx[folder_name]
                    for img_path in glob.glob(os.path.join(folder_path, "*.*")):
                        if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.JPG')):
                            self.image_paths.append(img_path)
                            self.labels.append(target_idx)
                            
        print(f"[Lab] Loaded {len(self.image_paths)} images from {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# =============================================
# Transforms
# =============================================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# TTA transforms (multiple augmented versions of same image)
tta_transforms = [
    test_transform, # Original
    transforms.Compose([ # Horizontal flip
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([ # Slight rotation
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([ # Slight zoom
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([ # Color adjusted
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
]

# =============================================
# Load Datasets
# =============================================
lab_train_dir = r"c:\Users\dltjr\Desktop\project\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\train"
outdoor_train_dir = r"c:\Users\dltjr\Desktop\project\plantdoc\train"
outdoor_test_dir = r"c:\Users\dltjr\Desktop\project\plantdoc\test"

# Load outdoor (plantdoc) dataset
outdoor_train_dataset = PlantDocMappedDataset(outdoor_train_dir, transform=train_transform)

# Load lab dataset and subsample to balance with outdoor data
# Ratio: ~2:1 (lab:outdoor) to maintain lab knowledge while emphasizing outdoor adaptation
lab_full_dataset = LabDataset(lab_train_dir, transform=train_transform)

# Subsample lab data to ~5000 images (roughly 2x outdoor data)
random.seed(42)
lab_sample_size = min(5000, len(lab_full_dataset))
lab_indices = random.sample(range(len(lab_full_dataset)), lab_sample_size)
lab_subset = Subset(lab_full_dataset, lab_indices)
print(f"[Lab Subset] Sampled {lab_sample_size} images from {len(lab_full_dataset)} total lab images")

# Combine datasets virtually (no files are moved!)
mixed_train_dataset = ConcatDataset([outdoor_train_dataset, lab_subset])
print(f"[Mixed] Total training images: {len(mixed_train_dataset)} (Outdoor: {len(outdoor_train_dataset)} + Lab: {lab_sample_size})")

# Test dataset (outdoor only - this is what we want to improve)
test_dataset = PlantDocMappedDataset(outdoor_test_dir, transform=test_transform)

mixed_train_loader = DataLoader(mixed_train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# =============================================
# Model Setup
# =============================================
model = PlantDiseaseEfficientNetB0(num_classes=38)
model.load_state_dict(torch.load('best_model_efficientnet_b0.pth', map_location=device))
model.to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# =============================================
# Stage 1: Freeze backbone, train classifier only (5 Epochs)
# =============================================
print("\n" + "="*60)
print("STAGE 1: Freeze Backbone, Train Classifier Only (5 Epochs)")
print("="*60)

# Freeze all backbone layers
for param in model.model.features.parameters():
    param.requires_grad = False

# Only optimize the classifier
optimizer_stage1 = optim.Adam(model.model.classifier.parameters(), lr=1e-3)
scheduler_stage1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage1, T_max=5)

stage1_epochs = 5
best_acc = 0.0
train_acc_history = []
val_acc_history = []
train_loss_history = []

for epoch in range(stage1_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    pbar = tqdm(mixed_train_loader, desc=f"S1 Epoch {epoch+1}/{stage1_epochs}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer_stage1.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_stage1.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        pbar.set_postfix({'Loss': loss.item(), 'Acc': 100 * correct_train / total_train})
    
    scheduler_stage1.step()
    epoch_train_loss = running_loss / len(mixed_train_loader)
    epoch_train_acc = 100 * correct_train / total_train
    train_loss_history.append(epoch_train_loss)
    train_acc_history.append(epoch_train_acc)
    
    # Evaluate
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    
    epoch_val_acc = 100 * correct_val / total_val
    val_acc_history.append(epoch_val_acc)
    print(f"S1 Epoch [{epoch+1}/{stage1_epochs}] Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, Val Acc: {epoch_val_acc:.2f}%")
    
    if epoch_val_acc > best_acc:
        best_acc = epoch_val_acc
        torch.save(model.state_dict(), 'finetuned_model_efficientnet_b0_v4.pth')
        print(f"  -> Best V4 model saved with Val Acc: {best_acc:.2f}%")

# =============================================
# Stage 2: Unfreeze all, fine-tune with very low LR (25 Epochs)
# =============================================
print("\n" + "="*60)
print("STAGE 2: Unfreeze All Layers, Fine-Tune Entire Model (25 Epochs)")
print("="*60)

# Unfreeze backbone
for param in model.model.features.parameters():
    param.requires_grad = True

# Very low LR for backbone, higher for classifier
optimizer_stage2 = optim.Adam([
    {'params': model.model.features.parameters(), 'lr': 1e-5},   # Backbone: very slow
    {'params': model.model.classifier.parameters(), 'lr': 5e-5},  # Classifier: moderate
], lr=1e-5)

stage2_epochs = 25
scheduler_stage2 = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage2, T_max=stage2_epochs)

for epoch in range(stage2_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    pbar = tqdm(mixed_train_loader, desc=f"S2 Epoch {epoch+1}/{stage2_epochs}")
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer_stage2.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_stage2.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        pbar.set_postfix({'Loss': loss.item(), 'Acc': 100 * correct_train / total_train})
    
    scheduler_stage2.step()
    epoch_train_loss = running_loss / len(mixed_train_loader)
    epoch_train_acc = 100 * correct_train / total_train
    train_loss_history.append(epoch_train_loss)
    train_acc_history.append(epoch_train_acc)
    
    # Evaluate
    model.eval()
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    
    epoch_val_acc = 100 * correct_val / total_val
    val_acc_history.append(epoch_val_acc)
    print(f"S2 Epoch [{epoch+1}/{stage2_epochs}] Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}%, Val Acc: {epoch_val_acc:.2f}%")
    
    if epoch_val_acc > best_acc:
        best_acc = epoch_val_acc
        torch.save(model.state_dict(), 'finetuned_model_efficientnet_b0_v4.pth')
        print(f"  -> Best V4 model saved with Val Acc: {best_acc:.2f}%")

print(f"\nV4 Fine-tuning complete (Both Stages). Best Validation Accuracy: {best_acc:.2f}%")

# =============================================
# Test-Time Augmentation (TTA) Evaluation
# =============================================
print("\n" + "="*60)
print("TEST-TIME AUGMENTATION (TTA) EVALUATION")
print("="*60)

# Reload best model
model.load_state_dict(torch.load('finetuned_model_efficientnet_b0_v4.pth', map_location=device))
model.to(device)
model.eval()

# TTA: For each image, predict with multiple augmented versions and average
test_dataset_raw = PlantDocMappedDataset(outdoor_test_dir, transform=None) # No transform, apply TTA manually

correct_tta = 0
total_tta = 0

with torch.no_grad():
    for idx in tqdm(range(len(test_dataset_raw)), desc="TTA Evaluation"):
        img_path = test_dataset_raw.image_paths[idx]
        label = test_dataset_raw.labels[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Apply all TTA transforms and collect predictions
        all_outputs = []
        for tta_t in tta_transforms:
            img_tensor = tta_t(image).unsqueeze(0).to(device)
            output = model(img_tensor)
            all_outputs.append(output)
        
        # Average all predictions (soft voting)
        avg_output = torch.mean(torch.stack(all_outputs), dim=0)
        _, predicted = torch.max(avg_output, 1)
        
        total_tta += 1
        if predicted.item() == label:
            correct_tta += 1

tta_acc = 100 * correct_tta / total_tta
print(f"\n[Without TTA] Best Val Acc: {best_acc:.2f}%")
print(f"[With TTA (5 augments)] Final Acc: {tta_acc:.2f}%")
print(f"[TTA Boost] +{tta_acc - best_acc:.2f}%p")

# =============================================
# Learning Curve Chart
# =============================================
total_epochs = stage1_epochs + stage2_epochs

plt.figure(figsize=(14, 5))

# Accuracy Chart
plt.subplot(1, 2, 1)
plt.plot(range(1, total_epochs + 1), train_acc_history, label='Train Accuracy', marker='o', markersize=3)
plt.plot(range(1, total_epochs + 1), val_acc_history, label='Validation Accuracy', marker='o', markersize=3)
plt.axvline(x=stage1_epochs, color='gray', linestyle='--', alpha=0.7, label='Stage 1→2 Transition')
plt.title('V4 Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.grid(True)

# Loss Chart
plt.subplot(1, 2, 2)
plt.plot(range(1, total_epochs + 1), train_loss_history, label='Train Loss', color='red', marker='o', markersize=3)
plt.axvline(x=stage1_epochs, color='gray', linestyle='--', alpha=0.7, label='Stage 1→2 Transition')
plt.title('V4 Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('learning_curve_v4.png', dpi=150)
print("Learning curve saved to 'learning_curve_v4.png'.")
