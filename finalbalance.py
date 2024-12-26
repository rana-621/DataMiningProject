import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


base_dir = 'chest_xray/chest_xray'
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')


def count_images(directory):
    class_counts = {}
    for class_name in os.listdir(directory):
        class_dir = os.path.join(directory, class_name)
        if os.path.isdir(class_dir):
            class_counts[class_name] = len(os.listdir(class_dir))
    return class_counts


print("Train distribution:", count_images(train_dir))
print("Validation distribution:", count_images(val_dir))
print("Test distribution:", count_images(test_dir))

#  تقسيم البيانات بالتساوي
def stratified_split_and_create_dirs(train_dir):
    pneumonia_images = [os.path.join(train_dir, 'PNEUMONIA', img) for img in os.listdir(os.path.join(train_dir, 'PNEUMONIA'))]
    normal_images = [os.path.join(train_dir, 'NORMAL', img) for img in os.listdir(os.path.join(train_dir, 'NORMAL'))]

    images = pneumonia_images + normal_images
    labels = [1] * len(pneumonia_images) + [0] * len(normal_images)

    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.4, stratify=labels)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp)

    base_balanced_dir = 'balanced_data'
    for subset, X, y in zip(['train', 'val', 'test'], [X_train, X_val, X_test], [y_train, y_val, y_test]):
        subset_dir = os.path.join(base_balanced_dir, subset)
        for img_path, label in zip(X, y):
            class_name = 'PNEUMONIA' if label == 1 else 'NORMAL'
            class_dir = os.path.join(subset_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            shutil.copy(img_path, class_dir)

stratified_split_and_create_dirs(train_dir)

  
balanced_train_dir = 'balanced_data/train'
balanced_val_dir = 'balanced_data/val'
balanced_test_dir = 'balanced_data/test'

  
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

   
train_generator = train_datagen.flow_from_directory(balanced_train_dir, target_size=(150, 150),
                                                    batch_size=32, class_mode='binary')
val_generator = val_datagen.flow_from_directory(balanced_val_dir, target_size=(150, 150),
                                                batch_size=32, class_mode='binary')

#  Build model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

 
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()


history = model.fit(train_generator, epochs=20, validation_data=val_generator)  # تم زيادة عدد الـ epochs إلى 20


model.save('FinalBalance_model.h5')  
 
test_generator = val_datagen.flow_from_directory(balanced_test_dir, target_size=(150, 150),
                                                 batch_size=32, class_mode='binary', shuffle=False)


test_images, test_labels = next(test_generator)
predictions = model.predict(test_images)
threshold = 0.5   
y_pred = (predictions > threshold).astype(int)

  
labels = [0, 1]  # Normal = 0, Pneumonia = 1
print(classification_report(test_labels, y_pred, target_names=['Normal', 'Pneumonia'], labels=labels))
print(confusion_matrix(test_labels, y_pred))
