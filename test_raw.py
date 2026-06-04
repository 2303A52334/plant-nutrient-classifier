import os
import numpy as np
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model('plant_nutrient_classifier.h5', compile=False)

def predict(image_path):
    img = Image.open(image_path).convert('RGB').resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array, verbose=0)
    return prediction

# Test a few from each
print("Testing Healthy Folder:")
for f in os.listdir('Healthy')[:5]:
    p = predict(os.path.join('Healthy', f))
    print(f"  {f}: {p}")

print("\nTesting Nutrient Folder:")
for f in os.listdir('Nutrient')[:5]:
    p = predict(os.path.join('Nutrient', f))
    print(f"  {f}: {p}")
