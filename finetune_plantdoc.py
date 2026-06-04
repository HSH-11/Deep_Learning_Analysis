import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import glob
from tqdm import tqdm

from model import PlantDiseaseEfficientNetB0


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Original 38 classes
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
                else:
                    print(f"Warning: Folder '{folder_name}' not found in original 38 classes.")
                    
        print(f"Loaded {len(self.image_paths)} images from {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# Transform with Augmentation for training
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 야외 환경 적응을 위한 빛/대비 증강
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load datasets
train_dir = os.path.join(PROJECT_DIR, "plantdoc", "train")
test_dir = os.path.join(PROJECT_DIR, "plantdoc", "test")

train_dataset = PlantDocMappedDataset(train_dir, transform=train_transform)
test_dataset = PlantDocMappedDataset(test_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

# Load Model
model = PlantDiseaseEfficientNetB0(num_classes=38)
model.load_state_dict(torch.load('best_model_efficientnet_b0.pth', map_location=device))
model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
# Fine-tuning: smaller learning rate so we don't destroy original weights
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training Loop
num_epochs = 5
best_acc = 19.12 # Base accuracy is 19.12%

print("Starting Fine-Tuning...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
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
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        pbar.set_postfix({'Loss': loss.item(), 'Acc': 100 * correct_train / total_train})
        
    train_acc = 100 * correct_train / total_train
    
    # Evaluation
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
            
    val_acc = 100 * correct_val / total_val
    print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), 'finetuned_model_efficientnet_b0.pth')
        print(f"  -> Best model saved with Val Acc: {best_acc:.2f}%")

print(f"Fine-tuning complete. Best Validation Accuracy: {best_acc:.2f}%")
