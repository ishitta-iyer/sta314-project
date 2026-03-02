# IMPORT LIBRARIES
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib

print("TensorFlow version:", tf.__version__)

# Load Data
train_directory = pathlib.Path('/data/classification-of-pet-facial-expression/train')
test_directory = pathlib.Path('/data/classification-of-pet-facial-expression/test')

class_names = sorted([d for d in os.listdir(train_directory) 
                      if os.path.isdir(os.path.join(train_directory, d))])
print(f"Found {len(class_names)} classes:")
print(class_names)

# Count images per class
print("\nImages per class:")
for class_name in class_names:
    class_path = os.path.join(train_directory, class_name)
    num_images = len([f for f in os.listdir(class_path) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))])
    print(f"{class_name}: {num_images} images")

# Display sample images
plt.figure(figsize=(12, 8))
for i, class_name in enumerate(class_names):  
    class_path = os.path.join(train_directory, class_name)
    img_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if img_files:
        img_path = os.path.join(class_path, img_files[0])
        img = plt.imread(img_path)
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        plt.title(class_name)
plt.tight_layout()
plt.show()


