# Skin Lesion Analyzer Web Application

This project is a web application designed for the analysis of skin lesion images using a machine learning model to predict potential skin conditions. It provides an interactive interface for users to upload lesion images, input patient details (name, gender, age, lesion location), and receive a generated report with a diagnosis and model confidence score. The application also maintains a history of patient records.

## Features

*   **Image-Based Prediction:** Utilizes a Convolutional Neural Network (CNN) model (e.g., EfficientNetB0 based) to classify skin lesions from uploaded images.
*   **Interactive Patient Data Input:** Allows input of patient name, and selection of gender, age (with +/- controls), and lesion location from predefined options.
*   **Report Generation:** Generates a textual report summarizing the diagnosis and patient information.
*   **Confidence Score & Visualization:** Displays the model's confidence in its prediction with a visual bar.
*   **Patient History:** Stores and allows viewing of past analysis records, searchable by patient name. Past records can be viewed in detail.
*   **User-Friendly Interface:** Web-based UI built with HTML, CSS, and vanilla JavaScript.

## Project Structure
skin-cancer-detector/
├── backend/
│ ├── model/ # Trained ML model and label encoder
│ │ ├── skin_lesion_classifier.keras
│ │ └── label_encoder.pkl
│ ├── static/
│ │ └── uploads/ # Stores user-uploaded lesion images (contents ignored by .gitignore)
│ ├── app.py # Flask backend server application
│ ├── patient_history.db # SQLite database for patient records (ignored by .gitignore)
│ └── schema.sql # Database schema definition
├── frontend/
│ ├── static/ # CSS, JS, and static images (icons) for the frontend
│ │ ├── doctor_icon.png
│ │ ├── history_icon.png
│ │ ├── style.css
│ │ └── script.js
│ └── index.html # Main HTML page for the UI
├── .gitignore # Specifies intentionally untracked files and folders
├── train_model.py # Python script to train the image classification model
├── requirements.txt # Python dependencies for the project
└── README.md # This file: Project overview and instructions

## Technologies Used

*   **Backend:**
    *   Python 3.x
    *   Flask (Web framework)
    *   TensorFlow & Keras (Machine learning model definition and training)
    *   Pillow (Image processing)
    *   NumPy (Numerical operations)
    *   Scikit-learn (Label encoding for model training)
    *   Imbalanced-learn (`imblearn` for handling class imbalance during model training)
    *   SQLite3 (Lightweight database for patient history)
*   **Frontend:**
    *   HTML5
    *   CSS3
    *   JavaScript (Vanilla JS for DOM manipulation, event handling, and API calls)
*   **Dataset (for training - example):**
    *   HAM10000 (The `train_model.py` script is set up for this dataset structure)

## Setup and Installation

Follow these steps to set up and run the project locally.

**1. Prerequisites:**

*   Python (version 3.7 - 3.10 recommended for TensorFlow compatibility, check your TF version)
*   `pip` (Python package installer)
*   Git (for cloning the repository)
*   (Optional but highly recommended) A virtual environment tool like `venv`.

**2. Clone the Repository:**

```bash
git clone https://github.com/saaif44/cancer-detection-app-NLP-flask
cd cancer-detection-app-NLP-flask

3. Create and Activate a Virtual Environment:
Navigate to the project root directory (skin-cancer-detector/).
Using venv (comes with Python):
# Create a virtual environment named 'venv'
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate


4. Install Dependencies:
Once your virtual environment is activated, install the required Python packages:
pip install -r requirements.txt


(This may take some time, especially for TensorFlow.)
5. Prepare the Machine Learning Model:
Dataset:
This project uses the HAM10000 dataset for training the example model in train_model.py.
Download the dataset (images and metadata).
Place HAM10000_metadata.csv, HAM10000_images_part_1/ (folder), and HAM10000_images_part_2/ (folder) into a known location.
Crucially, update the BASE_DATA_DIR variable at the top of train_model.py to point to the directory containing these dataset components.
Train the Model:
Run the training script from the project's root directory. This will generate skin_lesion_classifier.keras and label_encoder.pkl inside backend/model/.
python train_model.py

(This training process can be lengthy.)
Note: If you already have pre-trained model files (.keras and .pkl), you can skip this step and place them directly into the backend/model/ folder. Ensure the model architecture and preprocessing steps match those expected by backend/app.py.
6. Database Initialization:
The backend/app.py script will automatically attempt to initialize the SQLite database (patient_history.db) using backend/schema.sql if the database file doesn't already exist when the Flask app starts. Ensure backend/schema.sql is present.
Running the Application
Ensure your virtual environment is activated.
Navigate to the project's root directory (skin-cancer-detector/).
Start the Flask Backend Server:


python backend/app.py


You should see output indicating the server is running, typically on http://127.0.0.1:5000/. The terminal will also show if the model loaded correctly and if the database was initialized or found.
Access the Application:
Open your web browser and navigate to:
http://127.0.0.1:5000/
How to Use
Enter Patient Details: Fill in the patient's name. Select gender, adjust age using the controls, and choose the lesion location from the provided buttons.
Upload Lesion Image: Click the "Choose File" / "Lesion Image" button to select an image of the skin lesion. A preview of the selected image will appear.
Process: Click the "PROCESS" button.
View Report: The application will transition to an output page displaying the uploaded image, a generated diagnostic report, the predicted lesion type, and the model's confidence score.
View History: Click the history icon (top right of the header) to navigate to the patient history page.
You can view a list of past records.
Use the search bar to filter records by patient name.
Click the "Details" button on a history record to view its full report on the output page.
Click on a thumbnail in the history table to view a larger version of the image.
Troubleshooting
Model Not Found Error (Backend Startup):
Ensure python train_model.py has run successfully and that skin_lesion_classifier.keras and label_encoder.pkl are present in backend/model/.
Verify that MODEL_FILENAME and LABEL_ENCODER_FILENAME in backend/app.py match the actual filenames.
Static Files Not Loading (CSS/JS - Plain HTML):
Clear your browser cache or use "Disable cache" in browser developer tools (Network tab).
Verify the paths in frontend/index.html (e.g., href="frontend-static/style.css") and the corresponding route (@app.route('/frontend-static/...)) in backend/app.py.
Check the browser's developer console (F12) for 404 errors related to these files or other JavaScript errors.
Image Upload/Display Issues:
Check the browser console and Flask terminal for errors during the /predict request.
Ensure UPLOAD_FOLDER in backend/app.py is correctly configured and writable.
Verify that the static_url_path='/' is set correctly in the Flask app initialization in backend/app.py for serving uploaded images from the root URL.
preprocess_input Mismatch: The preprocess_input function used in backend/app.py (within preprocess_uploaded_image) must exactly match the one used during model training in train_model.py (e.g., from tensorflow.keras.applications.efficientnet import preprocess_input as model_specific_preprocess).
Future Enhancements
User authentication and distinct user accounts.
More sophisticated risk factor analysis based on patient data.
Allowing multiple image uploads per record.
Exporting reports to PDF.
Deployment to a cloud platform (e.g., Heroku, Render, AWS, Google Cloud).
Implementing Git LFS if model files become excessively large for standard Git tracking.
Contributing
(If you wish to accept contributions)
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change. Please ensure to update tests as appropriate and maintain code quality.
License
(Consider adding a license, e.g., MIT. If so, create a LICENSE.md file.)
This project is currently unlicensed. All rights reserved.