# 🚀 Heart Disease Prediction API

This project is an end-to-end machine learning service that predicts a patient's risk of heart disease. It includes data cleaning, exploratory data analysis (EDA), model training, and deployment as a **Docker** container, served by a **FastAPI** backend with a simple **web interface**.

-----

## ⚡ Live Demo

You can interact with the live, deployed service at the following URL:

➡️ **[https://heart-disease-predictor-25.onrender.com/](https://heart-disease-predictor-25.onrender.com/)** ⬅️

*(**Note:** This is a free-tier service on Render. It may "spin down" after a period of inactivity and take 30-60 seconds to "wake up" on the first visit.)*

-----

## 📋 Project Description & Approach

The goal of this project is to build a reliable service that can predict the presence of heart disease in a patient based on 13 key clinical features. The project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework, covering the entire ML lifecycle:


1.  **Business Understanding:** The problem is clear: heart disease is a leading cause of death. Early and accurate detection is critical. The goal is to create a model that assists medical professionals.
2.  **Data Understanding:** We use the UCI Heart Disease dataset. (See "Dataset" section).
3.  **Data Preparation:** This was a major part of the project. The raw data was messy and required significant cleaning and feature engineering.
4.  **Modeling:** Several models (Logistic Regression, Random Forest, XGBoost) were trained and evaluated.
5.  **Evaluation:** The models were evaluated with a focus on **Recall**, as minimizing false negatives is the top priority.
6.  **Deployment:** The final model, preprocessors, and API were packaged into a Docker container and deployed as a web service.

-----

## 💻 Tech Stack

  * **Experimentation:** `Jupyter Notebook`, `Pandas`, `Matplotlib`, `Seaborn`
  * **Environment:** `Pipenv` for dependency management
  * **Modeling:** `Scikit-learn`, `XGBoost`, `Numpy`
  * **API / Serving:** `FastAPI`, `Uvicorn`
  * **Containerization:** `Docker`
  * **Deployment:** `Render` (from Docker image)

-----

## 💾 Dataset

### Dataset Information

The model is trained on the **[UCI Cleveland Heart Disease dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)**, specifically from the `heart_disease_uci.csv` file. This file combines data from four different clinics, including the Cleveland Clinic Foundation.

### Dataset Features

The model is trained on the following 13 features:

| Feature | Description | Type |
| :--- | :--- | :--- |
| **age** | Patient's age in years | Numerical |
| **sex** | Biological sex of the patient | Categorical (Male/Female) |
| **cp** | Chest Pain Type | Categorical (Typical Angina, Atypical Angina, etc.) |
| **trestbps** | Resting Blood Pressure (in mm Hg) | Numerical |
| **chol** | Serum Cholesterol (in mg/dl) | Numerical |
| **fbs** | Fasting Blood Sugar \> 120 mg/dl? | Categorical (True/False) |
| **restecg** | Resting EKG Results | Categorical (Normal, ST-T Abnormality, etc.) |
| **thalch** | Maximum Heart Rate Achieved | Numerical |
| **exang** | Exercise Induced Angina? | Categorical (Yes/No) |
| **oldpeak** | ST Depression induced by exercise | Numerical |
| **slope** | Slope of the peak exercise ST segment | Categorical (Upsloping, Flat, etc.) |
| **ca** | Number of major vessels (0-3) colored by fluoroscopy | Numerical |
| **thal** | Thallium Stress Test Result | Categorical (Normal, Fixed Defect, etc.) |
| **target** | **Diagnosis (The Target)** | **0 (No Disease) / 1 (Has Disease)** |


### Dataset Issues

The raw dataset was a perfect example of a real-world data problem:

  * It contained **920 total records** from four sources.
  * It was **extremely "messy."** Most of the records (from outside the Cleveland clinic) were missing critical features. Columns like `slope`, `ca`, and `thal` had over 400-600 missing values.
  * The `target` variable (`num`) was not binary; it ranged from 0 (no disease) to 4.
  
-----

## 📊 Exploratory Data Analysis (EDA)

Initial analysis in `notebook.ipynb` revealed:

  * The raw dataset had many missing values, which were cleaned by dropping rows with any nulls in the 13 critical features, leaving the clean "Cleveland" subset.
  * The target variable (`target`) was fairly balanced, so complex class imbalance techniques were not required.
  * Categorical features like `cp` (Chest Pain Type) and `thal` (Thallium Test) showed a strong correlation with the target variable.


-----

## ⚙️ Project Execution

### Feature Engineering

Data preparation was the most critical step.

1.  **Data Cleaning:** Instead of complex imputation, we chose to filter for high-quality data. We dropped all rows that had *any* missing values in the 13 critical features. This effectively isolated the complete 297-record "Cleveland" subset, which is a standard benchmark.
2.  **Target Engineering:** The `num` column was converted into a binary `target` column: `0` (no disease) or `1` (has disease).
3.  **Feature Splitting:** Features were split into `numerical_features` (like `age`, `chol`) and `categorical_features` (like `sex`, `cp`).
4.  **Preprocessing:**
      * `StandardScaler` was fitted on the numerical features to normalize their range.
      * `DictVectorizer` was fitted on the categorical features. This was a key choice, as it allows our API to accept meaningful **strings** (like "Male", "Typical Angina") and automatically converts them into the one-hot encoded format the model expects.

### Model Training & Evaluation

To find the best model, three different classifiers were trained and evaluated on the validation set.

| Model | Recall (Validation) | AUC (Validation) |
| :--- | :---: | :---: |
| Logistic Regression | 0.7586 | 0.8509 |
| **Random Forest** | **0.7931** | 0.8676 |
| XGBoost | 0.7586 | 0.8699 |

### Key Challenge: Evaluation

The primary metric for this project was **Recall**.

In a medical diagnosis problem, a **False Negative** (predicting a sick patient is healthy) is far more dangerous than a **False Positive** (predicting a healthy patient is sick). `Recall` directly measures our model's ability to find all positive cases, making it the most important metric to optimize.

Based on this, the **Random Forest** was chosen as the champion model, as it had the highest Recall. It was then tuned using `RandomizedSearchCV` to find the optimal hyperparameters.

-----

## 📈 Results and Performance

The final, tuned Random Forest model (with `n_estimators=60`, `max_depth=5`, `min_samples_leaf=4`) was re-trained on the full 80% combined training/validation set and then evaluated one last time on the 20% hold-out **test set**.

**Final Model Performance (Test Set):**

| Metric | Score |
| :--- | :--- |
| **Recall** | **0.8400** |
| **AUC Score** | 0.9360 |
| **Accuracy** | 0.8833 |
| **Precision** | 0.8750 |
| **F1-Score** | 0.8571 |

### Key Findings

  * The final model is highly effective, correctly identifying 84% of all patients with heart disease in the test set.
  * The high AUC (0.936) indicates the model is also excellent at distinguishing between positive and negative cases.
  * The `DictVectorizer` workflow (training on strings) proved to be an excellent choice, as it simplifies the API logic in `predict.py` and allows the frontend to send human-readable JSON.

-----

## 🔧 How It's Configured to Run

### Model Training (`train.py`)

The `train.py` script is the "factory" for our model. When run, it:

1.  Loads the raw `heart_disease_uci.csv` data.
2.  Performs all the cleaning and feature engineering steps described above.
3.  Fits a `DictVectorizer` on the categorical string features.
4.  Fits a `StandardScaler` on the numerical features.
5.  Trains the final `RandomForestClassifier` with the best hyperparameters.
6.  Saves three critical artifacts to disk: `dv.pkl`, `scaler.pkl`, and `model.pkl`.

### Service Configuration (`predict.py` & `Dockerfile`)

The application is designed to be a self-contained, deployable unit.

1.  **`predict.py`**: This script uses **FastAPI** to create the web service.

      * At startup, it loads the three `.pkl` files (`dv.pkl`, `scaler.pkl`, `model.pkl`) into memory.
      * It serves the `index.html` webpage at the root URL (`/`).
      * It exposes the `/predict` endpoint, which accepts JSON data matching the `PatientData` Pydantic model (expecting strings for categorical features).
      * When a request comes in, it uses the loaded preprocessors to transform the data, feeds it to the model, and returns a JSON prediction.

2.  **`Dockerfile`**: This is the blueprint for our entire application.

      * It starts from a `python:3.11-slim` base image.
      * It installs all dependencies from `requirements.txt`.
      * It copies the `data/` and `train.py` script and **runs `train.py` inside the container**. This builds and saves the `.pkl` files directly into the image.
      * It copies the `predict.py` and `index.html` files.
      * It `EXPOSE`s port 8080 and sets the `CMD` to run the `uvicorn` server.

-----

## 📁 Repository Structure

```
heart-disease-api/
├── data/
│   └── heart_disease_uci.csv  # The raw dataset
├── .dockerignore                # Files for Docker to ignore during build
├── .gitignore                   # Files for Git to ignore
├── Dockerfile                   # Instructions to build the container
├── index.html                   # The HTML/CSS/JS frontend
├── notebook.ipynb               # The experimentation notebook (EDA, model selection)
├── Pipfile                      # Pipenv dependencies
├── Pipfile.lock                 # Pipenv exact dependencies
├── predict.py                   # The FastAPI application (serves API & frontend)
├── requirements.txt             # Frozen dependencies for Docker
└── train.py                     # Script to train and save the model & preprocessors
```

-----

## 🚀 Steps to Recreate & Run Locally

### 1\. Initial Setup (Experimentation)

(This is how the `notebook.ipynb` was run)

```bash
# Clone the repository
git clone https://github.com/owaisjunedi/heart-disease-predictor-25
cd heart-disease-api

# Install dependencies with pipenv
pipenv install jupyter pandas scikit-learn seaborn matplotlib xgboost fastapi uvicorn numpy

# Activate the virtual environment
pipenv shell

# Start the Jupyter notebook to explore
jupyter notebook
```

### 2\. Local API Test (Without Docker)

This shows how the scripts work together.

```bash
# From your active pipenv shell

# 1. Train the model and create the .pkl files
python train.py

# 2. In the same terminal, start the API server
python predict.py
# Server will be running at http://127.0.0.1:8080

# 3. Open a NEW terminal and activate the env
pipenv shell

# 4. Call the API with curl
curl -X POST "http://127.0.0.1:8080/predict" \
-H "Content-Type: application/json" \
-d '{
    "age": 63,
    "sex": "Male",
    "cp": "Typical Angina",
    "trestbps": 145,
    "chol": 233,
    "fbs": "True",
    "restecg": "Hypertrophy",
    "thalch": 150,
    "exang": "No",
    "oldpeak": 2.3,
    "slope": "Downsloping",
    "ca": 0.0,
    "thal": "Fixed Defect"
}'

# You will get a JSON response:
# {"heart_disease_probability":0.35969591005248025,"heart_disease_prediction":0}
```

### 3\. Local Deployment (With Docker)

This is the recommended way to run the project as it replicates the production environment. **You must have Docker Desktop running.**

```bash
# 1. Generate requirements.txt (if you haven't)
pipenv run pip freeze > requirements.txt

# 2. Build the Docker image
# This will run train.py inside the container
docker build -t heart-disease-api .

# 3. Run the container
docker run -p 8080:8080 heart-disease-api

# 4. Test the running container
# (From a new terminal)
curl -X POST "http://localhost:8080/predict" \
-H "Content-Type: application/json" \
-d '{
    "age": 63, "sex": "Male", "cp": "Typical Angina", "trestbps": 145,
    "chol": 233, "fbs": "True", "restecg": "Hypertrophy", "thalch": 150,
    "exang": "No", "oldpeak": 2.3, "slope": "Downsloping",
    "ca": 0.0, "thal": "Fixed Defect"
}'

# 5. Test the web interface
# Open your browser to http://localhost:8080
```

-----


## 🚀 Using the Application

Once the container is running, you can access it in two ways:

### 1\. Web Interface

Open your web browser and navigate to:
**[http://localhost:8080](https://www.google.com/search?q=http://localhost:8080)**

You will see the interactive web page and can submit predictions.

### 2\. API Endpoint (curl)

You can also send a `POST` request directly to the API endpoint.

```bash
curl -X POST "http://localhost:8080/predict" \
-H "Content-Type: application/json" \
-d '{
    "age": 63,
    "sex": "Male",
    "cp": "Typical Angina",
    "trestbps": 145,
    "chol": 233,
    "fbs": "True",
    "restecg": "Hypertrophy",
    "thalch": 150,
    "exang": "No",
    "oldpeak": 2.3,
    "slope": "Downsloping",
    "ca": 0.0,
    "thal": "Fixed Defect"
}'
```

**Expected Response:**

```json
{
  "heart_disease_probability": 0.35969591005248025,
  "heart_disease_prediction": 0
}
```

-----

## ☁️ Deployment

This service is deployed on **Render**.

The deployment is configured to connect to this GitHub repository. On every `git push`, Render automatically triggers a new build from the `Dockerfile`, trains the model, and deploys the new service.

You can test the live deployment endpoint here (replace with your URL):

```bash
curl -X POST "https://heart-disease-predictor-25.onrender.com/predict" \
-H "Content-Type: application/json" \
-d '{
    "age": 52,
    "sex": "Male",
    "cp": "Asymptomatic",
    "trestbps": 128,
    "chol": 204,
    "fbs": "True",
    "restecg": "Normal",
    "thalch": 156,
    "exang": "Yes",
    "oldpeak": 1.0,
    "slope": "Flat",
    "ca": 0.0,
    "thal": "Normal"
}'
```

-----


## 🔮 Future Scope & Improvements

  * **Advanced Imputation:** Instead of dropping 600+ records, use a more advanced imputation strategy (like KNN Imputer or MICE) to see if a model trained on the *full* dataset can perform better.
  * **More Tuning:** Use a more advanced tuning library like Optuna or Hyperopt to search a wider hyperparameter space.
  * **CI/CD Pipeline:** Implement a GitHub Actions workflow to automatically run tests, build the Docker image, and deploy to Render on every `git push` to the `main` branch.
  * **Model Monitoring:** Add basic monitoring (e.g., using Prometheus) to track model performance and data drift over time.

-----

## 📚 References

  * [DataTalksClub MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp)
  * [Zoomcamp Projects Showcase](https://datatalksclub-projects.streamlit.app/)
  * [UCI Machine Learning Repository: Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)