# Sleep Stage Prediction API

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![CatBoost](https://img.shields.io/badge/Model-CatBoost-orange)
![MySQL](https://img.shields.io/badge/Database-MySQL-4479A1?logo=mysql&logoColor=white)
![Status](https://img.shields.io/badge/Status-Development-yellow)

A machine learning-based sleep stage prediction system that uses wearable heart rate and motion data to predict sleep stages. The trained CatBoost model is integrated with FastAPI and MySQL to provide predictions through a REST API and store prediction records.

## Overview

This project develops an end-to-end machine learning pipeline for sleep stage prediction using physiological and movement data collected from a wrist-worn wearable device.

The system includes:

- Data preprocessing and feature engineering using Python.
- Sleep stage classification using CatBoost.
- Participant-wise model validation to evaluate generalization across individuals.
- FastAPI integration for real-time prediction requests.
- MySQL database integration for storing prediction records.
- Scalar for interactive API documentation.

## Dataset

**SleepAccel – Motion and heart rate from a wrist-worn wearable and labeled sleep from polysomnography (Version 1.0.0).**

- Source: [PhysioNet – SleepAccel](https://physionet.org/content/sleep-accel/1.0.0/)
- Wearable device: Apple Watch.
- Data includes heart rate, acceleration, step count and sleep-stage labels.
- Final selected dataset: 31 participants.
- Reference sleep-stage labels are obtained from polysomnography (PSG).

The raw dataset is processed to create features suitable for supervised machine learning.

## Machine Learning Pipeline

### 1. Data Preprocessing

- Load and process raw wearable and sleep-label data.
- Align physiological and motion measurements using timestamps.
- Convert observations into 30-second epochs.
- Handle missing values and prepare classification labels.

### 2. Feature Engineering

Features are extracted from heart rate and accelerometer measurements, including:

- Heart rate statistical features.
- Accelerometer X, Y and Z statistical features.
- Acceleration magnitude.
- Rolling-window statistics.
- Lag features.
- Motion-related features.

Rolling and lag features are constructed using past observations to avoid future-data leakage.

### 3. Model Training and Evaluation

- Algorithm: CatBoost Classifier.
- Framework: Scikit-learn and CatBoost.
- Validation: Stratified Group K-Fold.
- Grouping: Participant ID.
- Evaluation: Classification metrics and confusion matrix.

The model is evaluated using participant-wise validation to assess its ability to generalize to unseen participants.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Processing | Pandas, NumPy |
| Machine Learning | CatBoost, Scikit-learn |
| Model Persistence | Joblib |
| API | FastAPI |
| API Documentation | Scalar |
| Database | MySQL |
| Database ORM | SQLModel, SQLAlchemy |
| Environment Management | Python venv, python-dotenv |
| Development | Jupyter Notebook, VS Code, Git |

## Project Structure

```text
Sleep-Stage-Prediction/
│
├── app/
│   ├── __init__.py
│   ├── database.py
│   ├── db_models.py
│   ├── feature_engineering.py
│   ├── main.py
│   └── model.py
│
├── models/
│   ├── catboost_sleep_stage.cbm
│   └── catboost_feature_columns.pkl
│
├── notebooks/
│   └── sleep_stage_model.ipynb
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/kumarhimanshu7930-create/sleep-stage-prediction-project.git

cd sleep-stage-prediction-project
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows:**

```bash
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root directory.

```env
DB_USER=your_mysql_username
DB_PASSWORD=your_mysql_password
DB_HOST=localhost
DB_PORT=3306
DB_NAME=sleep_stage_db
```

Create the corresponding MySQL database before starting the API.

### 5. Start the API

```bash
fastapi dev app/main.py
```

## API Documentation

Once the server is running, access the interactive Scalar documentation:

```text
http://127.0.0.1:8000/scalar
```

### Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API home |
| POST | `/predict` | Predict sleep stages |
| GET | `/scalar` | Interactive API documentation |

### Prediction Input

The `/predict` endpoint accepts heart rate and motion measurements in JSON format.

Example:

```json
{
  "heart": [
    {
      "seconds": 30,
      "heart_rate_bpm": 82
    }
  ],
  "motion": [
    {
      "seconds": 30,
      "acceleration_x_g": 0.02,
      "acceleration_y_g": 0.01,
      "acceleration_z_g": 0.98
    }
  ]
}
```

The API processes the input, generates the required features, and returns the predicted sleep stages.

## Model Deployment

The trained CatBoost model is saved in `.cbm` format, while the corresponding feature-column order is stored using Joblib.

During API startup, the model and feature columns are loaded for inference.

The prediction pipeline:

1. Receives wearable sensor data.
2. Converts the input into DataFrames.
3. Performs feature engineering.
4. Generates sleep-stage predictions.
5. Stores prediction records in MySQL.
6. Returns predictions as a JSON response.

## Future Improvements

- Improve sleep-stage classification performance.
- Evaluate generalization across more participants.
- Develop continuous sleep monitoring.
- Integrate additional wearable sensor features.
- Develop a mobile application for sleep monitoring.

## Dataset Reference

SleepAccel: *Motion and heart rate from a wrist-worn wearable and labeled sleep from polysomnography*, Version 1.0.0. PhysioNet.

https://physionet.org/content/sleep-accel/1.0.0/

This project is intended for research and educational purposes and is not a medical diagnostic system.
