# Pneumonia Classifier

## Overview

This project develops a pneumonia classifier using PyTorch.  It analyzes chest X-ray images to predict whether a patient has pneumonia or does not. The project uses a pre-trained ResNet18 model, fine-tuned on a publicly available chest X-ray dataset, and deploys a Flask web application to provide an easy-to-use interface for uploading images and receiving predictions.

## Dataset

The dataset used is the "Chest X-Ray Images (Pneumonia)" dataset available on Kaggle: [https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).  This dataset is based on the paper "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning" by Kermany, D. S., Goldbaum, M., Cai, W., Valentim, C. C., Liang, H., Baxter, S. H., ... & Zhang, K. (2018). *Cell*, *172*(5), 1122-1131.

The dataset contains the following directory structure:

```
data/
│── chest_xray/
│   ├── test/
│   │   ├── NORMAL/       # Normal chest X-ray images for testing
│   │   └── PNEUMONIA/    # Pneumonia chest X-ray images for testing
│   │
│   ├── train/
│   │   ├── NORMAL/       # Normal chest X-ray images for training
│   │   └── PNEUMONIA/    # Pneumonia chest X-ray images for training
│   │
│   └── val/
│       ├── NORMAL/       # Normal chest X-ray images for validation
│       └── PNEUMONIA/    # Pneumonia chest X-ray images for validation

```


The dataset is divided into training, validation, and testing sets. Each set contains images categorized as either 'NORMAL' (no pneumonia) or 'PNEUMONIA' (presence of pneumonia).

## Project Structure

The project directory structure is as follows:
```
Pneumonia-Classifier
├── templates/            # Directory containing the HTML template for the web interface
│   └── index.html        # HTML file for the web interface
│── app.py                # Flask backend
│── train_model.py        # Model training script
│── requirements.txt      # Dependencies
│── Procfile              # Deployment config
│── pneumonia_classifier.pth # Trained model weights
│── README.md             # Project documentation

```



## Libraries Used

- torch
- torchvision
- Pillow (PIL)
- Flask
- scikit-learn
- requests

## Model Training and Architecture

1.  **Data Loading and Preprocessing:** The `PneumoniaDataset` class in `train_model.py` loads the images, applies transformations (resizing, conversion to tensor, normalization), and prepares them for training.
2.  **Model Architecture:** A ResNet18 model, pre-trained on ImageNet, is used as the base architecture. The final fully connected layer is modified to output predictions for two classes (NORMAL and PNEUMONIA).
3.  **Training Process:** The model is trained using the Adam optimizer and CrossEntropyLoss. The training loop iterates through the training data, calculates the loss, and updates the model's weights. The validation set is used to monitor the model's performance during training.
4.  **Training details:**
    *   **Optimizer:** Adam
    *   **Learning Rate:** 0.001
    *   **Loss Function:** CrossEntropyLoss
    *   **Epochs:** 10
    *   **Batch Size:** 64

## Web Application (Flask)

The Flask application (`app.py`) provides a web interface for users to upload X-ray images and receive predictions.

1.  **Image Upload:** Users can upload images through the web interface (`index.html`).
2.  **Prediction:** The uploaded image is preprocessed and fed into the trained ResNet18 model. The model outputs a prediction (NORMAL or PNEUMONIA).
3.  **Display:** The prediction is displayed to the user through the web interface.


https://github.com/user-attachments/assets/247ccc44-dfef-4c5f-83d3-52d7365e8a20


## Viewing the Web Interface

Run the Flask application locally by executing app.py, then open your browser at the specified local address to interact with the classifier.

## Future Improvements

*   Implement more advanced data augmentation techniques.
*   Experiment with different model architectures (e.g., ResNet50, EfficientNet).
*   Incorporate class weighting to address class imbalance.
*   Implement a learning rate scheduler.
*   Add logging and monitoring for the deployed application.
*   Provide probability outputs in the web interface, instead of just binary classifications.

## Author

Rafał Perfikowski

https://github.com/Rafal852

www.linkedin.com/in/rafal-perfikowski
