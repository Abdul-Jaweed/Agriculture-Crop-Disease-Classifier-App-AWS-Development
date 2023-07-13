from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

# Define class labels
class_names = [
    "Corn___Common_Rust",
    "Corn___Gray_Leaf_Spot",
    "Corn___Healthy",
    "Corn___Northern_Leaf_Blight",
    "Potato___Early_Blight",
    "Potato___Healthy",
    "Potato___Late_Blight",
    "Rice___Brown_Spot",
    "Rice___Healthy",
    "Rice___Leaf_Blast",
    "Rice___Neck_Blast",
    "Wheat___Brown_Rust",
    "Wheat___Healthy",
    "Wheat___Yellow_Rust"
]

# Load the model
model = tf.keras.models.load_model('model.h5')

# Initialize Flask app
app = Flask(__name__)

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for image classification
@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image from the request
    uploaded_file = request.files['image']
    
    # Preprocess the image
    image = Image.open(uploaded_file)
    image = image.resize((224, 224))  # Resize the image to match the model's input shape
    image = np.array(image)  # Convert image to numpy array
    image = image / 255.0  # Normalize the image
    image = np.reshape(image, (1, 224, 224, 3))  # Reshape the image to match the model's input shape
    
    # Perform prediction
    predictions = model.predict(image)
    
    # Get the predicted class and confidence
    class_index = np.argmax(predictions)
    class_name = class_names[class_index]
    confidence = predictions[0][class_index] * 100
    
    # Format the output
    output = {
        'Class': class_name.replace("___", " "),
        'Confidence': f"{confidence:.2f}%"
    }
    
    # Return the result as JSON
    return render_template('result.html', result=output)

# Run the Flask app
if __name__ == '__main__':
    app.run()
