#  Deep Learning-based Plant Leaf Disease Detection using CNN

This project detects diseases in plant leaves using a Convolutional Neural Network (CNN).
It is a supervised learning model trained on 3,824 images belonging to 16 classes.
It helps farmers and gardeners identify plant diseases from images, enabling faster and more accurate treatment.
The model is built using Python, TensorFlow, and Keras.

---

##  Features

1. Detects multiple leaf diseases from images
2. Provides predictions with high accuracy
3. Built using deep learning (CNN)
4. User-friendly web interface
5. Supports training from scratch and inference using a pre-trained model
6. Provides disease descriptions and cure information

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/ksayali16/leaf-disease-detector.git
cd leaf-disease-detector
```

### 2. Create and activate a virtual environment (Optional but recommended)

**Strictly use Python 3.9 for compatibility**

```bash
python -m venv leafdiseasedetectorenv
leafdiseasedetectorenv\Scripts\activate
```

### 3. Install required packages

```bash
pip install -r requirements.txt
```

Adjust package versions if needed to avoid dependency conflicts.

---

##  Dataset & Trained Model

Due to large file size, the dataset and trained model are hosted on Google Drive.

**Dataset:**  
https://drive.google.com/file/d/161nD6T_6UlIAof75is_eKppnBm9X2Z3V/view?usp=drive_link

**Trained Model (.h5):**  
https://drive.google.com/file/d/14WhhcBb6pg2gTqyIv-yvp7yOveAhylKa/view?usp=drive_link

---

##  Usage Options

This project supports two modes of usage:

###  Option 1: Train the model from scratch

```text
1. Split the dataset into training and validation sets using the script in src/
2. Update file paths in train.py
3. Run the training script
```

```bash
python src/train.py
```

Training may take 1–2 hours depending on your system.

---

###  Option 2: Use the pre-trained model (Recommended)

```text
1. Download the trained .h5 model
2. Update the model file path in app/app.py
3. Run the web application
```

```bash
python app/app.py
```

Once executed, the application will automatically open in your default web browser.

---

##  Testing & Evaluation

After obtaining the `.h5` file:

###  Prediction

```bash
python src/predict.py
```

###  Evaluation

```bash
python src/evaluate.py
```

Make sure to update file paths before running.

---

##  Code Overview

###  Model Training Logic

```text
train.py
- Loads dataset
- Builds CNN architecture
- Trains the model
- Saves trained .h5 file
```

###  Prediction Pipeline

```text
predict.py
- Loads trained model
- Preprocesses input image
- Predicts disease class
```

###  Evaluation Script

```text
evaluate.py
- Evaluates model on test dataset
- Generates accuracy metrics
```

###  Supporting Files

```text
split_data.py        → Dataset splitting
label_dict.py        → Class label mapping
description_dict.py  → Disease description & cure info
```

---

##  Project Structure

```text
leaf-disease-detector/
├── src/                      # Core ML logic
│   ├── train.py              # CNN training
│   ├── predict.py            # Inference script
│   ├── evaluate.py           # Model evaluation
│   ├── split_data.py         # Dataset preparation
│   ├── label_dict.py         # Class labels
│   └── description_dict.py   # Disease info
│
├── app/                      # Web interface + API
│   ├── app.py                # Flask app
│   └── index.html            # UI template
│
├── sampledataset/            # Small sample images
├── results/                  # Output screenshots
├── requirements.txt
└── README.md
```

---

##  Results

The `results/` folder contains:

- Confusion Matrix
- Accuracy Screenshot
- Sample Predictions

These demonstrate the performance of the trained model.

---

##  Future Improvements

1. Add more plant species
2. Improve accuracy using transfer learning
3. Mobile app integration
4. Real-time camera detection
5. Cloud deployment (AWS/GCP)

---

##  Author

Sayali Kurane  
Final-year CSE (Data Science) Student  
Aspiring AI/ML Engineer 🌿


