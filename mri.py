import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

# Load the model
model_path = "D:/ml/mri model/mrimodel.h5"  # Ensure the correct path format
model = load_model(model_path)

# Function to preprocess image for dementia detection
def preprocess_image(image_path):
    input_image = cv.imread(image_path)
    input_image_gs = cv.cvtColor(input_image, cv.COLOR_RGB2GRAY)
    input_image_gs = cv.resize(input_image_gs, (100, 100))
    input_image_gs = input_image_gs / 255.0  # Normalize to [0, 1]
    input_image_gs = input_image_gs.reshape(1, 100, 100, 1)  # Reshape for model input
    return input_image_gs

# Define class names
class_names = ['Benign', 'Glioma', 'Malignancy', 'Meningioma', 'No Tumor', 
               'Pituitary', 'adenocarcinoma', 'large-cell-carcinoma', 
               'normal', 'squamous-cell-carcinoma']


def predict_dementia(image_path):
    input_image = preprocess_image(image_path)
    prediction = model.predict(input_image)
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = class_names[predicted_class_index]
    return predicted_class_name
if __name__ == "__main__":
    import sys
    image_path = "D:\\ml\\mri model\\dataset\\test\\Pituitary\\Tr-pi_0090_jpg.rf.0f672ba9c9aec804996ad6757a32c093.jpg"
    predicted_class = predict_dementia(image_path)
    print(f"Predicted Class: {predicted_class}")
