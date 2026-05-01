# Bank Fraud Detection System

AI-powered fraud detection using **XGBoost**, **SHAP**, **Autoencoders**, and **Cost-Sensitive Learning**.

---

## 🧠 Technologies Used

| Technology | Purpose |
|------------|---------|
| **XGBoost** | Fraud classification |
| **SHAP** | Explain predictions |
| **Autoencoder** | Detect unusual patterns |
| **Cost-Sensitive Learning** | Penalize missed fraud |

---

## 📊 Results

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| XGBoost | 100% | 1.000 |
| Autoencoder | 99.1% | 0.996 |
| Ensemble | 93.0% | 0.999 |

- ✅ Zero missed fraud
- ✅ Real-time detection

---

## 🚀 How to Run

### Step 1: Clone & Setup
```bash
git clone https://github.com/anagha1507/bank-fraud-detection.git
cd bank-fraud-detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt

---

### Step 2: Train Models (One by One)
```bash
cd src
python data_preprocessing.py
python xgboost_model.py
python autoencoder_model.py
python shap_explainer.py
python final_evaluation.py

Step 3: Launch Web App
python app.py
Open browser → http://127.0.0.1:5000

Step 4: Detect Fraud
Option 1: Fill the form → Click "Detect Fraud"
Option 2: Click "Load Sample" buttons
Option 3: Upload CSV for batch detection