import tensorflow as tf
from PIL import Image
import numpy as np

def load_labels(label_file):
    with open(label_file, 'r') as f:
        return [line.strip() for line in f.readlines()]

def load_and_preprocess_image_keras(image_path, image_size):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(image_size)
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize
    return img_array
