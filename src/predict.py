from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = load_model(r"C:\Users\Admin\Documents\leaf disease-detector\leaf_classification_model.h5")

# Load the input image

img = load_img(r"dataset - Copy\dataset - Copy\test\test_tomatobacterialspot\0ab9c705-f29e-45ac-b786-9549b3c38f16___GCREC_Bact.Sp 3223_final_masked.jpg",target_size=(224, 224))

# Preprocess the input image
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

# Make predictions
prediction = model.predict(img_array)

# Interpret the results
predicted_class = np.argmax(prediction)
class_labels = ["bellpepperbacterialspot","bellpepperhealthy","cherryhealthy","cherrypowderymildew","grapeIsariopsis","grapeblackmeasles","grapeblackrot","grapehealthy","strawberryLeafscorch","strawberryhealthy","tomatoEarly_blight","tomatoLate_blight","tomatoLeaf_Mold","Septoria_leaf_spot","tomatobacterialspot","tomatohealthy","unidentified"]  # Replace with your actual class labels

print(f'Predicted class: {class_labels[predicted_class]}')
print(f'Confidence: {prediction[0][predicted_class]:.2f}')



