CREATE TABLE IF NOT EXISTS patient_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_name TEXT NOT NULL,
    age INTEGER,
    gender TEXT,
    lesion_location TEXT,
    image_filename TEXT,
    diagnosis_short TEXT,
    diagnosis_full TEXT,
    confidence REAL,
    report_text TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);