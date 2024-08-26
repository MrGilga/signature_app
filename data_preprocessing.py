import cv2
import os
import numpy as np
from pathlib import Path

current_file_directory = Path(__file__).parent
preprocessing_dir_path = Path(current_file_directory / "output" / "preprocessed_signatures")
os.makedirs(preprocessing_dir_path, exist_ok=True)

signature_path = Path(current_file_directory / "resources" / "datasets" / "cedar_signatures")
forge_dir = Path(signature_path, 'full_forg')
real_dir = Path(signature_path, 'full_org')

output_forge_dir = os.path.join(preprocessing_dir_path, 'preprocessed_forge')
output_real_dir = os.path.join(preprocessing_dir_path, 'preprocessed_real')

os.makedirs(output_forge_dir, exist_ok=True)
os.makedirs(output_real_dir, exist_ok=True)

target_size = (255, 255)

def preprocess_image(image_path, output_path, target_size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Impossibile caricare l'immagine: {image_path}")
        return
    
    old_size = image.shape[:2]
    
    ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
    new_size = tuple([int(x * ratio) for x in old_size])
    
    resized_image = cv2.resize(image, (new_size[1], new_size[0]))

    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    normalized_image = new_image / 255.0
    cv2.imwrite(output_path, normalized_image * 255)
    
    print(f"Immagine preprocessata salvata in: {output_path}")

if __name__ == "__main__":
    for filename in os.listdir(forge_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(forge_dir, filename)
            output_path = os.path.join(output_forge_dir, filename)
            preprocess_image(image_path, output_path, target_size)

    for filename in os.listdir(real_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(real_dir, filename)
            output_path = os.path.join(output_real_dir, filename)
            preprocess_image(image_path, output_path, target_size)
