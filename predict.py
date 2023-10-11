# predict.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('image_classifier_model.h5')

# Load and preprocess the image for prediction
img_path = 'path/to/your/image.jpg'
img = image.load_img(img_path, target_size=(150, 150))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Rescale to the same scale used during training

# Make predictions
predictions = model.predict(img_array)

# Print the prediction results
class_names = ['class1', 'class2']  # Replace with your class labels
predicted_class = np.argmax(predictions, axis=1)
predicted_label = class_names[predicted_class[0]]
print(f'Predicted class: {predicted_label}')
