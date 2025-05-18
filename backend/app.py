# backend/app.py
import os
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input 
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

import pickle
import uuid
import sqlite3
from datetime import datetime

# --- Configuration ---
# MODEL_FILENAME relative to the 'model' subdirectory within the backend
MODEL_FILENAME = 'skin_lesion_classifier.keras'
LABEL_ENCODER_FILENAME = 'label_encoder.pkl'
IMG_SIZE = 224
DATABASE_FILENAME = 'patient_history.db'
SCHEMA_FILENAME = 'schema.sql'


APP_ROOT = os.path.dirname(os.path.abspath(__file__)) # This is backend/
MODEL_DIR = os.path.join(APP_ROOT, 'model') # backend/model/
UPLOAD_FOLDER = os.path.join(APP_ROOT, 'static/uploads') # backend/static/uploads/
DATABASE_PATH = os.path.join(APP_ROOT, DATABASE_FILENAME) # backend/patient_history.db
SCHEMA_PATH = os.path.join(APP_ROOT, SCHEMA_FILENAME) # backend/schema.sql

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --- Initialize Flask App & Load Model ---
app = Flask(__name__,
            static_folder=UPLOAD_FOLDER, # Serves uploaded images from backend/static/uploads/ directly via /<filename>
            template_folder='../frontend',
            static_url_path='/' 
            ) # Looks for templates in ../frontend/ relative to backend/


# Route for frontend static assets (CSS, JS, icons)
@app.route('/frontend-static/<path:filename>')
def frontend_static(filename):
    # Serves files from ../frontend/static/ relative to backend/
    return send_from_directory(os.path.join(APP_ROOT, '../frontend/static'), filename)


MODEL_FULL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)
LABEL_ENCODER_FULL_PATH = os.path.join(MODEL_DIR, LABEL_ENCODER_FILENAME)

print(f"Attempting to load model from: {MODEL_FULL_PATH}")
print(f"Attempting to load label encoder from: {LABEL_ENCODER_FULL_PATH}")

model = None
label_encoder = None
CLASSES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc'] # Fallback

try:
    if not os.path.exists(MODEL_FULL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_FULL_PATH}")
    if not os.path.exists(LABEL_ENCODER_FULL_PATH):
        raise FileNotFoundError(f"Label encoder file not found at {LABEL_ENCODER_FULL_PATH}")

    model = load_model(MODEL_FULL_PATH)
    with open(LABEL_ENCODER_FULL_PATH, 'rb') as f:
        label_encoder = pickle.load(f)
    CLASSES = list(label_encoder.classes_)
    print("Model and LabelEncoder loaded successfully.")
    print(f"Classes: {CLASSES}")
except Exception as e:
    print(f"CRITICAL Error loading model or LabelEncoder: {e}")
    print("Ensure 'train_model.py' has been run successfully and model files are in backend/model/")


lesion_type_dict_full_names = {
    'nv': 'Melanocytic nevi (nv)',
    'mel': 'Melanoma (mel)',
    'bkl': 'Benign keratosis-like lesions (bkl)',
    'bcc': 'Basal cell carcinoma (bcc)',
    'akiec': 'Actinic keratoses (akiec)',
    'vasc': 'Vascular lesions (vasc)',
    'df': 'Dermatofibroma (df)'
}

# --- Database Setup ---
def get_db():
    db = sqlite3.connect(DATABASE_PATH)
    db.row_factory = sqlite3.Row
    return db

def init_db():
    if not os.path.exists(SCHEMA_PATH):
        print(f"ERROR: {SCHEMA_FILENAME} not found in {APP_ROOT}. Database cannot be initialized.")
        return
    if not os.path.exists(DATABASE_PATH): # Only init if DB file itself doesn't exist
        db = get_db()
        with open(SCHEMA_PATH, mode='r') as f:
            db.cursor().executescript(f.read())
        db.commit()
        db.close()
        print(f"Initialized the database at {DATABASE_PATH}.")

init_db()

# --- Helper Function for Image Preprocessing ---
def preprocess_uploaded_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        from tensorflow.keras.applications.efficientnet import preprocess_input as model_specific_preprocess
        img_array = model_specific_preprocess(img_array)
        return img_array
    except Exception as e:
        print(f"Error processing uploaded image {img_path}: {e}")
        return None

# --- Routes ---
@app.route('/')
def index():
    return send_from_directory('../frontend', 'index.html') # Serves from ../frontend/ relative to backend/

# Serve other frontend HTML files if any
@app.route('/<path:filename>.html')
def serve_html(filename):
    return send_from_directory('../frontend', f"{filename}.html")


@app.route('/predict', methods=['POST'])
def predict():
    if model is None or label_encoder is None:
        return jsonify({'error': 'Model not loaded. Check backend logs for CRITICAL errors.'}), 500

    if 'lesionImage' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['lesionImage']
    patient_name = request.form.get('patientName', 'N/A')
    gender = request.form.get('gender', 'unknown')
    age = request.form.get('age', 'unknown')
    lesion_location_input = request.form.get('lesionLocation', 'an unknown location')

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_ext = os.path.splitext(file.filename)[1]
        if not file_ext: # ensure there is an extension
            file_ext = '.jpg' # default if none
        img_filename = str(uuid.uuid4()) + file_ext
        img_path_on_server = os.path.join(UPLOAD_FOLDER, img_filename) # Full path to save
        file.save(img_path_on_server)

        processed_image = preprocess_uploaded_image(img_path_on_server)
        if processed_image is None:
            return jsonify({'error': 'Image preprocessing failed'}), 500

        prediction_output = model.predict(processed_image)
        predicted_class_idx = np.argmax(prediction_output, axis=1)[0]
        predicted_class_label = CLASSES[predicted_class_idx]
        confidence = float(np.max(prediction_output))
        predicted_class_full_name = lesion_type_dict_full_names.get(predicted_class_label, predicted_class_label)
        
        report_text = f"A Lesion diagnosed as {predicted_class_label} located on the {lesion_location_input} of a {gender} patient named {patient_name} aged {age}."
        report_diagnosis_confidence = f"Diagnosis: {predicted_class_full_name} (Confidence: {confidence:.2%})"

        try:
            db = get_db()
            db.execute(
                'INSERT INTO patient_records (patient_name, age, gender, lesion_location, image_filename, diagnosis_short, diagnosis_full, confidence, report_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (patient_name, age if age != 'unknown' else None, gender, lesion_location_input, img_filename, predicted_class_label, predicted_class_full_name, confidence, report_text)
            )
            db.commit()
            db.close()
        except Exception as e:
            print(f"Database error: {e}")

        # Image URL for frontend is just the filename, served by app.static_folder (UPLOAD_FOLDER)
        image_url_for_frontend = f'/{img_filename}' # e.g. /some-uuid.jpg

        return jsonify({
            'predicted_class_short': predicted_class_label,
            'predicted_class_full': predicted_class_full_name,
            'confidence': confidence,
            'report_text': report_text,
            'report_diagnosis_confidence': report_diagnosis_confidence,
            'image_url': image_url_for_frontend,
            'patient_name': patient_name
        })

    return jsonify({'error': 'An unexpected error occurred'}), 500

@app.route('/history', methods=['GET'])
def get_history():
    patient_name_query = request.args.get('patientName', '')
    db = get_db()
    if patient_name_query:
        records_cursor = db.execute(
            'SELECT * FROM patient_records WHERE patient_name LIKE ? ORDER BY timestamp DESC',
            ('%' + patient_name_query + '%',)
        )
    else:
        records_cursor = db.execute('SELECT * FROM patient_records ORDER BY timestamp DESC LIMIT 20')
    
    records = records_cursor.fetchall()
    db.close()
    
    # Convert image_filename to full URL for frontend
    processed_records = []
    for row_raw in records:
        row = dict(row_raw) # Convert sqlite3.Row to dict
        row['image_url'] = f"/{row['image_filename']}" # Served by app.static_folder
        processed_records.append(row)

    return jsonify(processed_records)


if __name__ == '__main__':
    # Ensure schema.sql is in backend/
    if not os.path.exists(SCHEMA_PATH):
        print(f"ERROR: {SCHEMA_FILENAME} not found at {SCHEMA_PATH}. Database cannot be initialized. Please create it.")
    else:
        init_db() # Call init_db to create DB if it doesn't exist
    app.run(debug=True, port=5000)