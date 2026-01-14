1. Deep Learning-based Plant Leaf Disease Detection using CNN

This project detects diseases in plant leaves using a Convolutional Neural Network (CNN).
It is a supervised learning model trained on 3,824 images belonging to 16 classes.
It helps farmers and gardeners identify plant diseases from images, enabling faster and more accurate treatment.
The model is built using Python, TensorFlow, and Keras.

2. Features

Detects multiple leaf diseases from images

Provides predictions with high accuracy

Built using deep learning (CNN)

User-friendly web interface

Supports training from scratch and inference using a pre-trained model

Provides disease descriptions and cure information

3. Installation
1. Clone the repository
git clone https://github.com/ksayali16/leaf-disease-detector.git
cd leaf-disease-detector

2. Create and activate a virtual environment (Optional but recommended)

# Strictly use Python 3.9 for compatibility

python -m venv leafdiseasedetectorenv
leafdiseasedetectorenv\Scripts\activate

3. Install required packages
pip install -r requirements.txt


Adjust package versions if needed to avoid dependency conflicts.

4. Dataset & Trained Model

Due to large file size, the dataset and trained model are hosted on Google Drive.

->Dataset: https://drive.google.com/file/d/161nD6T_6UlIAof75is_eKppnBm9X2Z3V/view?usp=drive_link

-> Trained Model (.h5): https://drive.google.com/file/d/14WhhcBb6pg2gTqyIv-yvp7yOveAhylKa/view?usp=drive_link

4. Usage Options

This project supports two modes of usage:

* Option 1: Train the model from scratch

    Split the dataset into training and validation sets using the script in src/.
    Update file paths in train.py.
    Run the training script: python src/train.py
    After training, a .h5 model file will be saved automatically.

# Training may take 1–2 hours depending on your system.

* Option 2: Use the pre-trained model (Recommended)
  
    Download the trained .h5 model from the link above.
    Update the model file path in app/app.py.
    Run the application:python app/app.py

# Once executed, the application will automatically open in your default web browser.

5. Testing & Evaluation

  After obtaining the .h5 file:
  
  For prediction:
  python src/predict.py
  
  For evaluation:
  python src/evaluate.py

# Make sure to update file paths before running.

📁 Project Structure
leaf-disease-detector/
├── src/                 # Core ML logic
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   ├── split_data.py
│   ├── label_dict.py
│   └── description_dict.py
│
├── app/                 # Web interface + API
│   ├── app.py
│   └── index.html
│
├── sampledataset/       # Small sample images
│
├── results/             # Screenshots of results
│
├── requirements.txt
└── README.md

**Results**

  The results/ folder contains:
    Confusion Matrix
    Accuracy Screenshot
    Sample Predictions
  These demonstrate the performance of the trained model.

**Future Improvements**
  1.Add more plant species
  2.Improve accuracy using transfer learning
  3.Mobile app integration
  4.Real-time camera detection
  5.Cloud deployment (AWS/GCP)

Author:
Sayali kurane
Final-year CSE (Data Science) Student
Aspiring AI/ML Engineer🌿 Deep Learning-based Plant Leaf Disease Detection using CNN


