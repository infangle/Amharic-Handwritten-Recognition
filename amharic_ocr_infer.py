import argparse
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMAGE_SIZE = (64, 64)  # Must match training image size

def load_label_map(label_map_path):
    with open(label_map_path, 'r') as f:
        label_to_index = json.load(f)
    index_to_label = {v: k for k, v in label_to_index.items()}
    return index_to_label

def preprocess_image(image_path):
    img = load_img(image_path, color_mode='grayscale', target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def predict_image(image_path, model_path='amharic_ocr_model.h5', label_map_path='label_to_index.json'):
    model = load_model(model_path)
    index_to_label = load_label_map(label_map_path)
    img_array = preprocess_image(image_path)
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions, axis=1)[0]
    predicted_label = index_to_label[predicted_index]
    return predicted_label

def main():
    parser = argparse.ArgumentParser(description='Amharic OCR Inference')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('--model', type=str, default='amharic_ocr_model.h5', help='Path to the trained model file')
    parser.add_argument('--label_map', type=str, default='label_to_index.json', help='Path to the label map JSON file')
    args = parser.parse_args()

    predicted_label = predict_image(args.image_path, args.model, args.label_map)
    print(f"Predicted Amharic character: {predicted_label}")

if __name__ == '__main__':
    main()
