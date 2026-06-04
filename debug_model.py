import os
import numpy as np
from PIL import Image
import keras

model = keras.models.load_model('plant_nutrient_classifier.h5', compile=False)
class_names = ['Healthy', 'Nutrient Deficiency']

def check(folder):
    print(f"\nChecking {folder}:")
    files = [f for f in os.listdir(folder) if f.endswith(('.jpg', '.jpeg', '.png'))][:5]
    for f in files:
        img_path = os.path.join(folder, f)
        img = Image.open(img_path).convert('RGB').resize((160, 160))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array, verbose=0)
        label_idx = np.argmax(pred)
        print(f"  {f}: Raw Pred={pred[0]}, Class={class_names[label_idx]} (Index={label_idx})")

check('Healthy')
check('Nutrient')
