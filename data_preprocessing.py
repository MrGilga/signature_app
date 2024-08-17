import cv2
import os
import numpy as np

project_dir = '/Users/dezso/Documents/GitHub/signature_app/'
forge_dir = os.path.join(project_dir, 'forge')
real_dir = os.path.join(project_dir, 'real')

output_forge_dir = os.path.join(project_dir, 'preprocessed_forge')
output_real_dir = os.path.join(project_dir, 'preprocessed_real')

os.makedirs(output_forge_dir, exist_ok=True)
os.makedirs(output_real_dir, exist_ok=True)

# target dimension
target_size = (255, 255)

def preprocess_image(image_path, output_path, target_size):
    # Carica l'immagine in scala di grigi
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print(f"Impossibile caricare l'immagine: {image_path}")
        return
    
    old_size = image.shape[:2]
    
    # Calcola il rapporto di ridimensionamento
    ratio = min(target_size[0] / old_size[0], target_size[1] / old_size[1])
    new_size = tuple([int(x * ratio) for x in old_size])
    
    # Ridimensiona l'immagine mantenendo le proporzioni
    resized_image = cv2.resize(image, (new_size[1], new_size[0]))
    
    # Aggiungi padding per raggiungere la dimensione target
    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    
    # Padding con colore nero
    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    # Normalizza l'immagine
    normalized_image = new_image / 255.0
    
    # Salva l'immagine preprocessata
    cv2.imwrite(output_path, normalized_image * 255)  # Moltiplica per 255 per salvare come immagine visibile
    
    print(f"Immagine preprocessata salvata in: {output_path}")

# Preprocessa tutte le immagini nella directory 'forge'
for filename in os.listdir(forge_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(forge_dir, filename)
        output_path = os.path.join(output_forge_dir, filename)
        preprocess_image(image_path, output_path, target_size)

# Preprocessa tutte le immagini nella directory 'real'
for filename in os.listdir(real_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(real_dir, filename)
        output_path = os.path.join(output_real_dir, filename)
        preprocess_image(image_path, output_path, target_size)
