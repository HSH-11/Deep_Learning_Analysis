import os
import glob
from rembg import remove
from PIL import Image
from tqdm import tqdm


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
base_input_dir = os.path.join(PROJECT_DIR, "plantdoc")
base_output_dir = os.path.join(PROJECT_DIR, "plantdoc_rembg")

# Find all images in train and test
all_images = []
for split in ['train', 'test']:
    input_dir = os.path.join(base_input_dir, split)
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(root, file))

print(f"Found {len(all_images)} images to process for V3.")

print("Starting background removal for entire dataset...")

for i, img_path in enumerate(tqdm(all_images)):
    # Create matching output directory structure
    rel_path = os.path.relpath(img_path, base_input_dir)
    out_path = os.path.join(base_output_dir, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # Change extension to .png since rembg outputs transparency (RGBA)
    out_path = os.path.splitext(out_path)[0] + ".png"
    
    if os.path.exists(out_path):
        continue
        
    try:
        input_image = Image.open(img_path).convert("RGBA")
        output_image = remove(input_image)
        output_image.save(out_path)
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

print("V3 Background removal completed for all images.")
