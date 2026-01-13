from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  # Import CORS
import tensorflow as tf
from PIL import Image
import io
import numpy as np
from classlabels import class_labels 
from cure import disease_cures


app = Flask(__name__,template_folder='./')
CORS(app)  # Enable CORS globally

# Load your trained leaf disease model
model = tf.keras.models.load_model(r"C:\Users\Admin\Documents\leaf disease-detector\leaf_classification_model.h5")

def preprocess_image(image):
    image = image.resize((224, 224))  # Resize for model input
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

def get_disease_cure(predicted_label):
    disease_name = predicted_label.lower().strip()
    print("Predicted Label:", disease_name)
    for key, cure in disease_cures.items():
        print("Checking Key:", key.lower().strip())
        if disease_name == key.lower().strip():
            print("Match Found!")
            return cure
    print("No Match Found!")
    return "No cure available for this disease."

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_file = request.files["image"]
    image = Image.open(io.BytesIO(image_file.read()))
    processed_image = preprocess_image(image)

    # Make prediction
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction)  # Get predicted class index

    # Get label from CLASS_LABELS dictionary
    predicted_label = class_labels.get(predicted_class, "Unknown Disease")
    predicted_label = str(predicted_label) 

    # Get disease cure
    cure = get_disease_cure(predicted_label)
    
    return jsonify({
        "prediction": predicted_label,
        "cure": cure
    })  # Return the label instead of a class numbers
if __name__ == "__main__":
    app.run(debug=True)
    