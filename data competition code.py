import nibabel as nib
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Load NIfTI image
def load_nii(file_path, dtype=np.float64):
    nii = nib.load(file_path)
    data = nii.get_fdata(dtype=dtype)
    return data

# Normalize images
def normalize_images(images):
    normalized_images = []
    for img in images:
        if np.max(img) > 0:  # Avoid division by zero
            normalized_images.append(img / np.max(img))
        else:
            normalized_images.append(img)
    return normalized_images

# Data preprocessing: resize to target shape
def preprocess_data(images, labels, target_shape=(128, 128, 64)):
    images_resized = []
    labels_resized = []

    for img in images:
        img_resized = tf.image.resize(img, target_shape[:2]).numpy()
        img_resized = tf.image.resize(img_resized.transpose(2, 0, 1), (target_shape[2], target_shape[0])).numpy()
        images_resized.append(img_resized.transpose(1, 2, 0))

    for lbl in labels:
        lbl_resized = tf.image.resize(lbl, target_shape[:2]).numpy()
        lbl_resized = tf.image.resize(lbl_resized.transpose(2, 0, 1), (target_shape[2], target_shape[0])).numpy()
        labels_resized.append(lbl_resized.transpose(1, 2, 0))

    return np.array(images_resized), np.array(labels_resized)

# Define 3D U-Net model
def unet_model_3d(input_shape):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    conv1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling3D((2, 2, 2))(conv1)

    conv2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling3D((2, 2, 2))(conv2)

    # Decoder
    up1 = layers.UpSampling3D((2, 2, 2))(pool2)
    conv3 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up1)
    conv3 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv3)

    up2 = layers.UpSampling3D((2, 2, 2))(conv3)
    conv4 = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(up2)
    conv4 = layers.Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv4)

    outputs = layers.Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')(conv4)

    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load training data and labels
image_folder = "D:/BaiduNetdiskDownload/imagesTr"
label_folder = "D:/BaiduNetdiskDownload/labelsTr"

image_files = sorted([os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith('.nii')])
label_files = sorted([os.path.join(label_folder, f) for f in os.listdir(label_folder) if f.endswith('.nii')])

images = [load_nii(f, dtype=np.float16) for f in image_files]
labels = [load_nii(f, dtype=np.float16) for f in label_files]

# Normalize data
images = normalize_images(images)
labels = normalize_images(labels)

# Split into training and validation sets
images_train, images_val, labels_train, labels_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Data preprocessing
target_shape = (128, 128, 64)  # Target shape
images_train, labels_train = preprocess_data(images_train, labels_train, target_shape)
images_val, labels_val = preprocess_data(images_val, labels_val, target_shape)

# Check data shapes
print(f"Training images shape: {images_train.shape}")
print(f"Training labels shape: {labels_train.shape}")
print(f"Validation images shape: {images_val.shape}")
print(f"Validation labels shape: {labels_val.shape}")

# Define the model
model = unet_model_3d((128, 128, 64, 1))

# Compile and train the model
model.fit(
    images_train[..., np.newaxis],  # Add channel dimension
    labels_train[..., np.newaxis],  # Add channel dimension
    validation_data=(images_val[..., np.newaxis], labels_val[..., np.newaxis]),
    epochs=50,
    batch_size=4
)

# Load test data
new_images_folder = "D:/BaiduNetdiskDownload/imagesTs"
new_image_files = sorted([os.path.join(new_images_folder, f) for f in os.listdir(new_images_folder) if f.endswith('.nii')])

new_images = [load_nii(f, dtype=np.float32) for f in new_image_files]
new_images = normalize_images(new_images)

# Data preprocessing
new_images_processed, _ = preprocess_data(new_images, new_images, target_shape)

# Predict results
predictions = model.predict(new_images_processed[..., np.newaxis])

# Binarize predictions
binary_predictions = (predictions > 0.5).astype(np.uint8)

# Save predictions as NIfTI files
output_folder = "D:/BaiduNetdiskDownload/prediction"
os.makedirs(output_folder, exist_ok=True)

for i, pred in enumerate(binary_predictions):
    output_path = os.path.join(output_folder, f"prediction_{i}.nii")
    nib.save(nib.Nifti1Image(pred[..., 0], np.eye(4)), output_path)  # Remove channel dimension

print("Prediction saved successfully!")