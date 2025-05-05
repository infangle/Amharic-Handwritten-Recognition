import os
import re
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow as tf

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Constants
DATA_DIRS = ['data/Amharic Character Dataset 1', 'data/Amharic Character Dataset 2']
IMAGE_SIZE = (64, 64)  # Resize images to 64x64
BATCH_SIZE = 32
EPOCHS = 2
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

def extract_label(filename):
    # Extract label prefix before the first dot, e.g. '001he' from '001he.1.jpg'
    match = re.match(r'([^.]+)\.', filename)
    if match:
        return match.group(1)
    else:
        return None

def load_dataset():
    images = []
    labels = []
    for data_dir in DATA_DIRS:
        for filename in os.listdir(data_dir):
            if filename.lower().endswith('.jpg'):
                label = extract_label(filename)
                if label is None:
                    continue
                img_path = os.path.join(data_dir, filename)
                # Load image in grayscale mode
                img = load_img(img_path, color_mode='grayscale', target_size=IMAGE_SIZE)
                img_array = img_to_array(img) / 255.0  # Normalize to [0,1]
                images.append(img_array)
                labels.append(label)
    return np.array(images), np.array(labels)

def encode_labels(labels):
    unique_labels = sorted(set(labels))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    indices = np.array([label_to_index[label] for label in labels])
    categorical_labels = to_categorical(indices, num_classes=len(unique_labels))
    return categorical_labels, label_to_index

def build_model(num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    print("Loading dataset...")
    images, labels = load_dataset()
    print(f"Loaded {len(images)} images.")

    print("Encoding labels...")
    categorical_labels, label_to_index = encode_labels(labels)
    print(f"Number of classes: {len(label_to_index)}")

    # Split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        images, categorical_labels, test_size=TEST_SIZE, random_state=42, stratify=categorical_labels)

    # Further split train+val into train and val
    val_size_adjusted = VALIDATION_SIZE / (1 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size_adjusted, random_state=42, stratify=y_train_val)

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")

    # Build model
    model = build_model(num_classes=len(label_to_index))
    model.summary()

    # Train model with early stopping
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=[early_stop]
    )

    # Evaluate on test set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc:.4f}")

    # Save model and label mapping
    model.save('amharic_ocr_model.h5')
    print("Model saved to amharic_ocr_model.h5")

    # Save label mapping to a file
    import json
    with open('label_to_index.json', 'w') as f:
        json.dump(label_to_index, f)
    print("Label mapping saved to label_to_index.json")

def predict_image(image_path, model_path='amharic_ocr_model.h5', label_map_path='label_to_index.json'):
    from tensorflow.keras.models import load_model
    import json
    from tensorflow.keras.preprocessing.image import load_img, img_to_array
    import numpy as np

    # Load model and label map
    model = load_model(model_path)
    with open(label_map_path, 'r') as f:
        label_to_index = json.load(f)
    index_to_label = {v: k for k, v in label_to_index.items()}

    # Load and preprocess image
    img = load_img(image_path, color_mode='grayscale', target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_label = index_to_label[predicted_index]
    return predicted_label

if __name__ == '__main__':
    main()

    # Test the trained model on a sample image
    sample_image_path = 'path/to/sample_image.jpg'  # TODO: Update this path to your test image
    predicted_label = predict_image(sample_image_path)
    print(f"Predicted label for the sample image: {predicted_label}")
