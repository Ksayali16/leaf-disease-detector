from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
MODEL_PATH = "leaf_classification_model.h5"
VAL_DIR = r"C:\Users\Admin\Documents\leaf disease-detector\dataset - Copy\dataset - Copy\val"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Load model
model = load_model(MODEL_PATH)

# Data generator
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Get true labels
true_labels = val_generator.classes
class_names = list(val_generator.class_indices.keys())

# Predict
predictions = model.predict(val_generator)
predicted_labels = np.argmax(predictions, axis=1)

# Accuracy
accuracy = np.mean(predicted_labels == true_labels)
print(f"\nValidation Accuracy: {accuracy:.4f}\n")

# Classification Report
print("Classification Report:\n")
print(classification_report(true_labels, predicted_labels, target_names=class_names))

# Confusion Matrix
cm = confusion_matrix(true_labels, predicted_labels)

plt.figure(figsize=(20, 16))  # bigger figure so all labels fit
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names, cbar=True)
plt.xticks(rotation=45, ha='right', fontsize=5)  # rotate x-axis labels and reduce font
plt.yticks(rotation=0, fontsize=5)               # y-axis labels horizontal
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()  # adjust layout so nothing is cut off
plt.show()
