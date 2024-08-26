import cv2
import os
import numpy as np
from pathlib import Path
import config

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
    # forged signatures
    for filename in os.listdir(config.test_dataset_config.test_dataset_2_forge_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(config.test_dataset_config.test_dataset_2_forge_dir, filename)
            output_path = os.path.join(config.test_dataset_config.test_dataset_2_output_forge_dir, filename)
            preprocess_image(image_path, output_path, target_size)

    # original signatures
    for filename in os.listdir(config.test_dataset_config.test_dataset_2_real_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(config.test_dataset_config.test_dataset_2_real_dir, filename)
            output_path = os.path.join(config.test_dataset_config.test_dataset_2_output_real_dir, filename)
            preprocess_image(image_path, output_path, target_size)
