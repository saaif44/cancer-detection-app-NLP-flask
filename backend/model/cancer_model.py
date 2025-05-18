# train_model.py
import os
import pandas as pd
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.over_sampling import RandomOverSampler
import pickle

# --- Configuration ---
BASE_DATA_DIR = r"\dataset" 
METADATA_FILE = os.path.join(BASE_DATA_DIR, 'HAM10000_metadata.csv')
IMAGE_DIR_PART1 = os.path.join(BASE_DATA_DIR, 'HAM10000_images_part_1')
IMAGE_DIR_PART2 = os.path.join(BASE_DATA_DIR, 'HAM10000_images_part_2')

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 4 # More is not Efficient, less is not good, 20 is safe
MODEL_SAVE_PATH = 'backend/model/skin_lesion_classifier.keras' # Using .keras format
LABEL_ENCODER_PATH = 'backend/model/label_encoder.pkl'

# Ensure model directory exists
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

# --- 1. Load Data and Prepare Paths ---
print("Loading metadata...")
data = pd.read_csv(METADATA_FILE)

print("Creating image path dictionary...")
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(IMAGE_DIR_PART1, '*.jpg'))}
imageid_path_dict.update({os.path.splitext(os.path.basename(x))[0]: x
                          for x in glob(os.path.join(IMAGE_DIR_PART2, '*.jpg'))})

data['path'] = data['image_id'].map(imageid_path_dict.get)
data = data.dropna(subset=['path']) # Remove rows where image path is missing

# Map lesion types (optional, but good for understanding)
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}
data['cell_type'] = data['dx'].map(lesion_type_dict.get)
data['cell_type_idx'] = pd.Categorical(data['dx']).codes # For stratification

# --- 2. Image Preprocessing Function ---
def preprocess_image_for_training(path):
    try:
        img = image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = preprocess_input(img_array) # Specific to EfficientNet
        return img_array
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        return None

print("Preprocessing images (this may take a while)...")
# Create a temporary column for image arrays to allow for easier oversampling
# processing images after splitting and oversampling to save memory initially for large datasets
# For this dataset size, processing all images first is feasible but can be memory intensive.
# Let's process paths, then handle images during oversampler or batch generation if memory is an issue.
# For simplicity here, if RAM allows, preprocess all. Otherwise, a generator approach is better.

# Filter out images that couldn't be processed (if any)
# data['image_array'] = data['path'].apply(preprocess_image_for_training)
# data = data.dropna(subset=['image_array'])
# image_data_np = np.stack(data['image_array'].values)


# --- 3. Label Encoding ---
print("Encoding labels...")
labels = data['dx'].values
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(labels)
y = to_categorical(integer_encoded)

# Save the label encoder
with open(LABEL_ENCODER_PATH, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"LabelEncoder saved to {LABEL_ENCODER_PATH}")
print(f"Classes found by LabelEncoder: {list(label_encoder.classes_)}")


# --- 4. Data Splitting (using paths first) ---
# We need to load and preprocess images for X_train, X_test specifically after splitting paths
print("Splitting data...")
train_paths, test_paths, y_train, y_test = train_test_split(
    data['path'].values, y,
    test_size=0.2, random_state=42, stratify=y # Stratify by original y for better distribution
)

train_idx_for_oversampling, val_idx_for_oversampling, _, _ = train_test_split(
    range(len(train_paths)), train_paths, # Dummy second arg for splitting indices
    test_size=0.2, random_state=42, stratify=y_train # Stratify again for validation set
)

# Get actual paths for train and validation
actual_train_paths = train_paths[train_idx_for_oversampling]
actual_val_paths = train_paths[val_idx_for_oversampling]
y_train_actual = y_train[train_idx_for_oversampling]
y_val_actual = y_train[val_idx_for_oversampling]


print(f"Initial train samples: {len(actual_train_paths)}, Validation samples: {len(actual_val_paths)}, Test samples: {len(test_paths)}")

# --- 5. Load and Preprocess Images for Train/Val/Test Sets and Oversample Train ---
print("Loading and preprocessing images for train/val/test sets...")

X_train_img = np.array([preprocess_image_for_training(path) for path in actual_train_paths if preprocess_image_for_training(path) is not None])
y_train_actual = y_train_actual[[preprocess_image_for_training(path) is not None for path in actual_train_paths]] # Align y with valid images

X_val_img = np.array([preprocess_image_for_training(path) for path in actual_val_paths if preprocess_image_for_training(path) is not None])
y_val_actual = y_val_actual[[preprocess_image_for_training(path) is not None for path in actual_val_paths]]

X_test_img = np.array([preprocess_image_for_training(path) for path in test_paths if preprocess_image_for_training(path) is not None])
y_test = y_test[[preprocess_image_for_training(path) is not None for path in test_paths]]


print(f"Processed train samples: {len(X_train_img)}, Val samples: {len(X_val_img)}, Test samples: {len(X_test_img)}")

# --- 6. Oversampling (on training data only) ---
# RandomOverSampler expects 2D data for X, so we reshape image data
print("Applying RandomOverSampler to training data...")
nsamples, nx, ny, nz = X_train_img.shape
X_train_reshaped = X_train_img.reshape((nsamples, nx*ny*nz))

# We need to oversample based on 1D integer labels, not one-hot encoded y_train_actual
y_train_integers = np.argmax(y_train_actual, axis=1)

oversampler = RandomOverSampler(random_state=42)
X_train_resampled_flat, y_train_resampled_integers = oversampler.fit_resample(X_train_reshaped, y_train_integers)

# Reshape X_train_resampled back to image format
X_train_resampled = X_train_resampled_flat.reshape(-1, nx, ny, nz)
# Convert y_train_resampled back to one-hot encoding
y_train_resampled = to_categorical(y_train_resampled_integers, num_classes=y_train_actual.shape[1])

print(f"Original training shape: {X_train_img.shape}, {y_train_actual.shape}")
print(f"Resampled training shape: {X_train_resampled.shape}, {y_train_resampled.shape}")


# --- 7. Model Definition ---
print("Defining the model...")
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
base_model.trainable = True # Fine-tune some layers or all

# Fine-tuning: unfreeze layers towards the end of the base model
# Example: Unfreeze top 20 layers (adjust based on model architecture)
# for layer in base_model.layers[:-20]:
#    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x) # Add dropout for regularization
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
predictions = Dense(y_train_resampled.shape[1], activation='softmax')(x) # num_classes

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', # Consider AdamW or a lower learning rate for fine-tuning
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 8. Callbacks ---
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# --- 9. Model Training ---
print("Starting model training...")
history = model.fit(
    X_train_resampled, y_train_resampled,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(X_val_img, y_val_actual),
    callbacks=[early_stopping, reduce_lr]
)

# --- 10. Save Model ---
print(f"Saving model to {MODEL_SAVE_PATH}...")
model.save(MODEL_SAVE_PATH)
print("Model saved successfully.")

# --- 11. Evaluate Model (Optional here, but good practice) ---
print("Evaluating model on test set...")
loss, accuracy = model.evaluate(X_test_img, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

print("Training script finished.")