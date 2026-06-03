import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import glob
import cv2

from model import PlantDiseaseEfficientNetB0
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Original 38 classes (sorted alphabetically as ImageFolder did originally)
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
idx_to_class = {i: cls_name for i, cls_name in enumerate(ORIGINAL_CLASSES)}

# Transform
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load model
model = PlantDiseaseEfficientNetB0(num_classes=38)
model.load_state_dict(torch.load('best_model_efficientnet_b0.pth', map_location=device))
model.to(device)
model.eval()

# Load all images
test_dir = r"c:\Users\dltjr\Desktop\project\plantdoc\test"
image_paths = []
labels = []
for folder_name in os.listdir(test_dir):
    folder_path = os.path.join(test_dir, folder_name)
    if os.path.isdir(folder_path) and folder_name in class_to_idx:
        target_idx = class_to_idx[folder_name]
        for img_path in glob.glob(os.path.join(folder_path, "*.*")):
            if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(img_path)
                labels.append(target_idx)

# Output directory
output_dir = "gradcam_results"
os.makedirs(output_dir, exist_ok=True)

# Target layer for Grad-CAM in EfficientNet-B0
target_layers = [model.model.features[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

wrong_predictions_count = 0
max_images = 5

for img_path, label_idx in zip(image_paths, labels):
    if wrong_predictions_count >= max_images:
        break
        
    rgb_img_pil = Image.open(img_path).convert("RGB")
    rgb_img_resized = rgb_img_pil.resize((224, 224))
    rgb_img_float = np.float32(rgb_img_resized) / 255
    
    input_tensor = test_transform(rgb_img_pil).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()
        
    if pred_idx != label_idx:
        # Generate Grad-CAM for the predicted class
        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]
        
        cam_image = show_cam_on_image(rgb_img_float, grayscale_cam, use_rgb=True)
        
        # Save image
        true_name = idx_to_class[label_idx]
        pred_name = idx_to_class[pred_idx]
        out_filename = f"wrong_{wrong_predictions_count + 1}_true_{true_name}_pred_{pred_name}.png"
        out_path = os.path.join(output_dir, out_filename)
        
        # Convert RGB to BGR for cv2
        cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        
        # Add text
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cam_image_bgr, f"True: {true_name[:15]}", (10, 20), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(cam_image_bgr, f"Pred: {pred_name[:15]}", (10, 40), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        cv2.imwrite(out_path, cam_image_bgr)
        wrong_predictions_count += 1
        print(f"Saved {out_filename}")

print(f"Finished generating {wrong_predictions_count} Grad-CAM images in {output_dir}/")
