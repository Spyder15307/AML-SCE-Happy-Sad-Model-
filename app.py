from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64


app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
model = load_model("models/happy_sad_model.h5")

# Define allowed extensions for uploaded files
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Function to check if the file extension is allowed
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/insert-image', methods=['GET', 'POST'])
def insert_image():
    if request.method == 'POST':
        # Check if a file is provided
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser should submit an empty file without a filename
        if file.filename == '':
            return redirect(request.url)
        # If the file exists and has an allowed extension
        if file and allowed_file(file.filename):
            # Save the file securely
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Load the image
            img = cv2.imread(filepath)
            # Resize the image
            img_resized = cv2.resize(img, (256, 256))
            # Preprocess the image for prediction
            img_processed = img_resized / 255.0
            img_processed = np.expand_dims(img_processed, axis=0)
            # Predict using the model
            prediction = model.predict(img_processed)
            # Determine the emotion based on the prediction
            emotion = "Happy" if prediction < 0.5 else "Sad"
            # emotion = str(predic)
            pre = str(prediction)
            # Encode image to base64
            _, img_encoded = cv2.imencode('.jpg', img)
            img_base64 = base64.b64encode(img_encoded).decode('utf-8')
            # Pass the filename, emotion, and base64 encoded image to the result template
            return render_template('result.html', emotion=emotion, prediction=pre, img_base64=img_base64)
    return render_template('insert_image.html')

@app.route('/live-webcam')
def live_webcam():
    return render_template('live_webcam.html')



@app.route('/result', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        if 'photo' in request.files:
            photo = request.files['photo']
            # Save the photo
            photo.save('static/captured_photo.jpg')
        
        return 'No photo uploaded.'
    
    elif request.method == 'GET':                    
        img = cv2.imread("static\captured_photo.jpg")
        # Resize the image
        img_resized = cv2.resize(img, (256, 256))
        # Preprocess the image for prediction
        img_processed = img_resized / 255.0
        img_processed = np.expand_dims(img_processed, axis=0)
        # Predict using the model
        prob = model.predict(img_processed)
        # Determine the emotion based on the prediction
        emotion = "Happy" if prob < 0.8 else "Sad"
        # emotion = str(predict)
        pre = str(prob)
        # Encode image to base64
        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        # Pass the filename, emotion, and base64 encoded image to the result template
        return render_template('result.html', emotion=emotion, prediction=pre, img_base64=img_base64)

if __name__ == '__main__':
    app.run(debug=True)
