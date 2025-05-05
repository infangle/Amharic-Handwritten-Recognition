import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

IMAGE_SIZE = (64, 64)  # Must match training image size

class AmharicOCRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Amharic Handwritten Character Recognition")

        self.model_path = 'amharic_ocr_model.h5'
        self.label_map_path = 'label_to_index.json'

        self.model = load_model(self.model_path)
        self.index_to_label = self.load_label_map(self.label_map_path)

        self.image_path = None

        # GUI Elements
        self.select_button = tk.Button(root, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=10)

        self.image_label = tk.Label(root, text="No image selected")
        self.image_label.pack()

        self.predict_button = tk.Button(root, text="Predict Character", command=self.predict_character, state=tk.DISABLED)
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 16))
        self.result_label.pack(pady=10)

        self.image_display = tk.Label(root)
        self.image_display.pack()

    def load_label_map(self, label_map_path):
        with open(label_map_path, 'r') as f:
            label_to_index = json.load(f)
        return {v: k for k, v in label_to_index.items()}

    def select_image(self):
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.image_path = file_path
            self.image_label.config(text=f"Selected: {file_path}")
            self.predict_button.config(state=tk.NORMAL)
            self.display_image(file_path)

    def display_image(self, image_path):
        img = Image.open(image_path)
        img.thumbnail((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        self.image_display.config(image=img_tk)
        self.image_display.image = img_tk  # Keep reference

    def preprocess_image(self, image_path):
        img = load_img(image_path, color_mode='grayscale', target_size=IMAGE_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array

    def predict_character(self):
        if not self.image_path:
            messagebox.showwarning("No Image", "Please select an image first.")
            return
        img_array = self.preprocess_image(self.image_path)
        predictions = self.model.predict(img_array)
        predicted_index = np.argmax(predictions, axis=1)[0]
        predicted_label = self.index_to_label[predicted_index]
        self.result_label.config(text=f"Predicted Amharic Character: {predicted_label}")

def main():
    root = tk.Tk()
    app = AmharicOCRApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
