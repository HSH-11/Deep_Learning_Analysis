"""Visualize ViT CLS-token attention on PlantDoc test samples (correct + wrong)."""
import glob
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import PlantDiseaseViTSmall

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
idx_to_class = {i: cls_name for i, cls_name in enumerate(ORIGINAL_CLASSES)}

test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def get_cls_attention_map(model, input_tensor):
    """Extract mean CLS-to-patch attention from the last ViT block."""
    vit = model.model
    x = vit.patch_embed(input_tensor)
    cls_token = vit.cls_token.expand(x.shape[0], -1, -1)
    if vit.cls_token is not None:
        x = torch.cat((cls_token, x), dim=1)
    x = x + vit.pos_embed
    x = vit.pos_drop(x)

    for block in vit.blocks[:-1]:
        x = block(x)

    last_block = vit.blocks[-1]
    norm_x = last_block.norm1(x)
    attn_module = last_block.attn
    batch_size, num_tokens, channels = norm_x.shape
    qkv = (
        attn_module.qkv(norm_x)
        .reshape(batch_size, num_tokens, 3, attn_module.num_heads, channels // attn_module.num_heads)
        .permute(2, 0, 3, 1, 4)
    )
    query, key, _ = qkv[0], qkv[1], qkv[2]
    attn = (query @ key.transpose(-2, -1)) * attn_module.scale
    attn = attn.softmax(dim=-1)
    cls_attn = attn.mean(dim=1)[0, 0, 1:]
    grid_size = int((cls_attn.shape[0]) ** 0.5)
    return cls_attn.reshape(grid_size, grid_size).detach().cpu().numpy()


def overlay_attention(rgb_img, attn_map):
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    attn_resized = cv2.resize(attn_map, (rgb_img.shape[1], rgb_img.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = np.uint8(0.55 * rgb_img + 0.45 * heatmap)
    return overlay


checkpoint = os.path.join(PROJECT_DIR, "finetuned_model_vit_small_v2.pth")
output_dir = os.path.join(PROJECT_DIR, "vit_attention_results")
os.makedirs(output_dir, exist_ok=True)

model = PlantDiseaseViTSmall(num_classes=38, pretrained=False).to(device)
model.load_state_dict(torch.load(checkpoint, map_location=device))
model.eval()

test_dir = os.path.join(PROJECT_DIR, "plantdoc", "test")
image_paths = []
labels = []
for folder_name in os.listdir(test_dir):
    folder_path = os.path.join(test_dir, folder_name)
    if os.path.isdir(folder_path) and folder_name in class_to_idx:
        target_idx = class_to_idx[folder_name]
        for img_path in glob.glob(os.path.join(folder_path, "*.*")):
            if img_path.lower().endswith((".png", ".jpg", ".jpeg")):
                image_paths.append(img_path)
                labels.append(target_idx)

def save_attention_image(rgb_pil, input_tensor, label_idx, pred_idx, prefix, index):
    rgb_resized = rgb_pil.resize((224, 224))
    rgb_float = np.asarray(rgb_resized, dtype=np.float32)
    attn_map = get_cls_attention_map(model, input_tensor)
    overlay = overlay_attention(rgb_float, attn_map)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    true_name = idx_to_class[label_idx]
    pred_name = idx_to_class[pred_idx]
    filename = f"{prefix}_{index}_true_{true_name}_pred_{pred_name}.png"
    color = (0, 255, 0) if prefix == "correct" else (0, 0, 255)
    cv2.putText(
        overlay_bgr,
        f"True: {true_name[:20]}",
        (10, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        overlay_bgr,
        f"Pred: {pred_name[:20]}",
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.imwrite(os.path.join(output_dir, filename), overlay_bgr)
    print(f"Saved {filename}")


max_images = 5
correct_saved = 0
wrong_saved = 0
correct_total = 0
wrong_total = 0

for img_path, label_idx in zip(image_paths, labels):
    rgb_pil = Image.open(img_path).convert("RGB")
    input_tensor = test_transform(rgb_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred_idx = output.argmax(dim=1).item()

    is_correct = pred_idx == label_idx
    if is_correct:
        correct_total += 1
        if correct_saved < max_images:
            save_attention_image(rgb_pil, input_tensor, label_idx, pred_idx, "correct", correct_saved + 1)
            correct_saved += 1
    else:
        wrong_total += 1
        if wrong_saved < max_images:
            save_attention_image(rgb_pil, input_tensor, label_idx, pred_idx, "wrong", wrong_saved + 1)
            wrong_saved += 1

    if correct_saved >= max_images and wrong_saved >= max_images:
        break

print(
    f"\nTest set: {correct_total} correct / {wrong_total} wrong "
    f"({100 * correct_total / len(image_paths):.2f}% accuracy)"
)
print(f"Saved {correct_saved} correct + {wrong_saved} wrong attention images in {output_dir}/")
