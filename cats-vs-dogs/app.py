import os
import numpy as np
import cv2
import pickle
from flask import Flask, request, render_template, redirect, url_for
from skimage.feature import hog

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Ensure upload directory exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load the trained model
with open('cats-vs-dogs/data1.pickle', 'rb') as f:
    model = pickle.load(f)

# Define categories
categories = ['Cat', 'Dog']

# Function to extract HOG features
def extract_features(image):
    features, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    return features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        # Save the uploaded file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        # Preprocess the image
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img_resized = cv2.resize(img, (100, 100))
        features = extract_features(img_resized).reshape(1, -1)

        # Predict the label using the SVM model
        label_index = model.predict(features)[0]
        category = categories[label_index]

        return render_template('index.html', prediction=f"The image is classified as: {category}", image=file.filename)

if __name__ == '__main__':
    app.run(debug=True)
