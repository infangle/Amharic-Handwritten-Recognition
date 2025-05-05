# Amharic Handwritten Character Recognition

This project implements a handwritten Amharic character recognition system using deep learning. It includes scripts for training a convolutional neural network (CNN) model, running inference on images, and a graphical user interface (GUI) for easy interaction.

## Project Description

- **train_amharic_ocr.py**: Loads Amharic handwritten character datasets, preprocesses images, builds and trains a CNN model, evaluates it, and saves the trained model and label mappings.
- **amharic_ocr_infer.py**: A command-line script to load the trained model and predict the Amharic character from a given image.
- **amharic_ocr_gui.py**: A Tkinter-based GUI application that allows users to select an image and get the predicted Amharic character interactively.
- **amharic_prefix_to_char_map.py**: A script to map dataset file name prefixes to actual Amharic characters in a 34 x 7 grid format.

## Future Improvements

- Increase the number of training epochs and augment the dataset for better accuracy.
- Implement more advanced CNN architectures or transfer learning for improved performance.
- Add support for recognizing Amharic words or sentences, not just individual characters.
- Enhance the GUI with additional features like batch processing and confidence scores.
- Provide a web-based interface for easier accessibility.

## How to Clone and Run the Project

1. Clone the repository:

```bash
git clone https://github.com/infangle/Amharic-Handwritten-Recognition.git
cd Amharic-Handwritten-Recognition
```

2. (Optional) Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required dependencies:

```bash
pip install -r requirements.txt
```

4. Train the model:

```bash
python train_amharic_ocr.py
```

5. Run inference on an image via command line:

```bash
python amharic_ocr_infer.py path/to/image.jpg
```

6. Launch the GUI application:

```bash
python amharic_ocr_gui.py
```

## Notes

- Update the sample image path in the inference script or GUI as needed.
- Ensure the trained model file `amharic_ocr_model.h5` and label mapping `label_to_index.json` are present in the project directory before running inference or GUI.
