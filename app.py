from flask import Flask, render_template, request, flash, redirect, url_for
import os
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import tensorflow as tf
import keras
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "plant_nutrient_secret_key"  # Required for flash messages

UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

# Load model once at startup
try:
    model = keras.models.load_model('plant_nutrient_classifier.h5', compile=False)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

class_names = ['Healthy', 'Nutrient Deficiency']

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    if model is None:
        return "Model not loaded", 0.0
    
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)
        confidence = float(prediction[0][predicted_label])
        
        return class_names[predicted_label], confidence
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "Error during prediction", 0.0

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                prediction, confidence = predict_image(file_path)
                
                # Make image_path relative for template
                relative_image_path = file_path.replace(os.sep, '/')
                
                return render_template('index.html', 
                                     prediction=prediction, 
                                     confidence=confidence, 
                                     image_path=relative_image_path)
            except Exception as e:
                logger.error(f"Upload error: {e}")
                flash('An error occurred during upload.')
                return redirect(request.url)
        else:
            flash('Invalid file type. Allowed: png, jpg, jpeg')
            return redirect(request.url)
            
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
