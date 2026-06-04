import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

# Constants - Optimized for lower RAM and faster CPU training
IMG_SIZE = 160
BATCH_SIZE = 16

def load_images_from_folder(folder_path, label):
    images, labels = [], []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            try:
                img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
                images.append(np.array(img))
                labels.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    return images, labels

# Load data
print("Loading images...")
images0, labels0 = load_images_from_folder('Healthy', 0)
images1, labels1 = load_images_from_folder('Nutrient', 1)

X = np.array(images0 + images1) / 255.0
y = np.array(labels0 + labels1)
y_cat = to_categorical(y, num_classes=2)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.15, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

# Calculate class weights
y_ints = np.argmax(y_train, axis=1)
weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_ints), y=y_ints)
class_weights = dict(enumerate(weights))

# Base model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = False

# New head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

print("Starting training phase 1...")
model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
          epochs=5,
          validation_data=(X_val, y_val),
          class_weight=class_weights)

print("Starting training phase 2 (fine-tuning)...")
base_model.trainable = True
# Freeze first 120 layers, tune the rest
for layer in base_model.layers[:120]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
          epochs=5,
          validation_data=(X_val, y_val),
          class_weight=class_weights,
          callbacks=[early_stop])

model.save("plant_nutrient_classifier.h5")
print("Model saved successfully.")

y_pred = np.argmax(model.predict(X_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print("\nAccuracy Test on Sample Data:")
print(classification_report(y_true, y_pred, target_names=['Healthy', 'Nutrient']))
