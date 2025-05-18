# 🧴 Skin Lesion Analyzer Web Application

A web-based skin lesion analysis tool using deep learning. This application allows users to upload skin lesion images, fill in basic patient details, and receive a diagnosis prediction with a confidence score. The tool also keeps track of past patient records.

---

## 🚀 Features

- **🖼️ Image-Based Prediction:** Uses a CNN model (EfficientNetB0) to classify uploaded skin lesion images.
- **📝 Patient Data Input:** Enter patient name, gender, age (with +/-), and lesion location.
- **📄 Report Generation:** Auto-generated report with prediction and confidence.
- **📊 Confidence Score:** Visual bar representation of the model's confidence.
- **📁 Patient History:** Viewable/searchable record of past predictions.
- **🌐 User-Friendly Interface:** Built with HTML, CSS, and vanilla JavaScript.

---

## 📁 Project Structure
```
skin-cancer-detector/
├── backend/
│ ├── model/
│ │ ├── skin_lesion_classifier.keras
│ │ └── label_encoder.pkl
│ ├── static/uploads/ # Uploaded lesion images
│ ├── app.py # Flask backend
│ ├── schema.sql # DB schema
│ └── patient_history.db # SQLite database (auto-generated)
├── frontend/
│ ├── static/
│ │ ├── style.css
│ │ ├── script.js
│ │ ├── doctor_icon.png
│ │ └── history_icon.png
│ └── index.html # Main interface
├── train_model.py # ML model training script
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md # This file
```

---

## 🛠️ Technologies Used

### Backend:
- Python 3.x
- Flask
- TensorFlow + Keras
- Pillow
- NumPy
- scikit-learn
- imbalanced-learn
- SQLite3

### Frontend:
- HTML5
- CSS3
- JavaScript (Vanilla)

---

## 📦 Dataset Used

- **HAM10000**: Human Against Machine with 10000 training images.  
  Download required metadata and image folders:
  - `HAM10000_metadata.csv`
  - `HAM10000_images_part_1/`
  - `HAM10000_images_part_2/`

---

## ⚙️ Setup Instructions

### 1. Prerequisites
- Python 3.7 – 3.10
- `pip`
- Git
- (Optional) `venv` for virtual environment

### 2. Clone the Repository
```bash
git clone https://github.com/saaif44/cancer-detection-app-NLP-flask
cd cancer-detection-app-NLP-flask
```
3. Create and Activate Virtual Environment
# Windows
```bash
# install python 3.10 if not already installed
py -3.10 -m venv venv
venv\Scripts\activate
pip install --upgrade pip
pip install tensorflow
```
4. Install Dependencies
```
pip install -r requirements.txt
```
5. Train the ML Model (Optional)
Update BASE_DATA_DIR in train_model.py to point to your dataset path, then:
```
#optional because we already have the trained model
python train_model.py
```
This will generate:

backend/model/skin_lesion_classifier.keras

backend/model/label_encoder.pkl

OR manually place pre-trained model files in the backend/model/ folder.

🧪 Run the Application
1. Start the Flask Server
```
python backend/app.py
```
2. Open in Browser
Go to: http://127.0.0.1:5000/

🧑‍⚕️ How to Use
Enter Patient Info – Name, gender, age, lesion location.

Upload Image – Upload lesion photo (preview appears).

Click PROCESS – Model predicts lesion type.

View Report – See full prediction, image, and confidence score.

History Page – See/search past records and open full reports.

🧯 Troubleshooting
Model Not Loading?
Check if skin_lesion_classifier.keras and label_encoder.pkl exist.

Paths in backend/app.py should match filenames.

Static Files Not Displaying?
Clear browser cache or disable cache via DevTools.

Ensure correct route mappings in Flask for /frontend-static.

Image Upload Issue?
Check Flask terminal and browser console logs.

Ensure UPLOAD_FOLDER in Flask app is correctly set.

Mismatched preprocess_input?
Ensure the same preprocessing function is used in both train_model.py and app.py.

🌱 Future Enhancements
🔐 User login system

📊 Risk analysis based on medical history

🖼️ Multiple image upload support

🧾 Export PDF report

☁️ Cloud deployment (Render/Heroku/AWS)

📁 Git LFS for large model files

🤝 Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss your ideas. Ensure code quality and update relevant files when contributing.
