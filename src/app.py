"""
Bank Fraud Detection Web Application
Flask backend with REST API
"""
import os
import sys
sys.path.append('.')

import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap

app = Flask(__name__, template_folder='../templates', static_folder='../static')
CORS(app)

# Global variables for models
xgb_model = None
autoencoder_model = None
xgb_threshold = 0.5
ae_threshold = 0.5
feature_names = None
explainer = None

def load_models():
    """Load all trained models."""
    global xgb_model, autoencoder_model, xgb_threshold, ae_threshold, feature_names, explainer
    
    print("Loading models...")
    
    # Load XGBoost
    with open('../models/xgboost_fraud_model.pkl', 'rb') as f:
        xgb_data = pickle.load(f)
    xgb_model = xgb_data['model']
    xgb_threshold = xgb_data.get('optimal_threshold', 0.5)
    feature_names = xgb_data['feature_names']
    
    # Load Autoencoder
    autoencoder_model = tf.keras.models.load_model('../models/autoencoder_model/autoencoder.keras')
    with open('../models/autoencoder_model/metadata.pkl', 'rb') as f:
        ae_data = pickle.load(f)
    ae_threshold = ae_data['threshold']
    
    # Create SHAP explainer
    explainer = shap.TreeExplainer(xgb_model)
    
    print(f"✅ Models loaded! XGB threshold: {xgb_threshold:.4f}, AE threshold: {ae_threshold:.6f}")


def preprocess_transaction(transaction_dict):
    """Preprocess a single transaction."""
    df = pd.DataFrame([transaction_dict])
    
    # Feature engineering
    df['balance_change_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
    df['errorBalanceOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['errorBalanceDest'] = df['oldbalanceDest'] - df['newbalanceDest']
    df['amount_to_old_balance'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['amount_to_new_balance'] = df['amount'] / (df['newbalanceOrig'] + 1)
    df['balance_ratio'] = df['oldbalanceOrg'] / (df['oldbalanceDest'] + 1)
    df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['amount_log'] = np.log1p(df['amount'])
    df['high_amount'] = (df['amount'] > 10000).astype(int)
    df['risk_score'] = (
        (df['high_amount'] * 0.3) +
        (df['is_night'] * 0.1) +
        (df['is_weekend'] * 0.05) +
        ((df['txn_count_24h'] > 10).astype(int) * 0.2) +
        ((df['location_distance'] > 100).astype(int) * 0.2) +
        ((df['previous_frauds'] > 0).astype(int) * 0.15)
    )
    
    # Encode categoricals
    type_map = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4}
    df['type'] = df['type'].map(type_map).fillna(0)
    
    gender_map = {'M': 0, 'F': 1}
    df['gender'] = df['gender'].map(gender_map).fillna(0)
    
    device_map = {'Mobile': 0, 'Web': 1, 'ATM': 2, 'POS': 3}
    df['device'] = df['device'].map(device_map).fillna(0)
    
    # Add missing columns
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    
    df = df[feature_names]
    return df


def predict_single(transaction_dict):
    """Predict fraud for a single transaction."""
    features = preprocess_transaction(transaction_dict)
    features_array = features.values
    
    # XGBoost prediction
    xgb_proba = xgb_model.predict_proba(features_array)[0, 1]
    xgb_pred = xgb_proba >= xgb_threshold
    
    # Autoencoder prediction
    reconstructed = autoencoder_model.predict(features_array, verbose=0)
    mse = np.mean(np.square(features_array - reconstructed), axis=1)[0]
    ae_score = mse / ae_threshold
    ae_pred = mse > ae_threshold
    
    # Ensemble
    ae_normalized = 1 / (1 + np.exp(-(ae_score - 1) * 5))
    ensemble_score = (0.6 * xgb_proba) + (0.4 * ae_normalized)
    ensemble_pred = ensemble_score >= 0.5
    
    # Risk level
    if ensemble_score >= 0.8:
        risk_level = "CRITICAL"
        risk_color = "#dc3545"
    elif ensemble_score >= 0.6:
        risk_level = "HIGH"
        risk_color = "#fd7e14"
    elif ensemble_score >= 0.4:
        risk_level = "MEDIUM"
        risk_color = "#ffc107"
    else:
        risk_level = "LOW"
        risk_color = "#28a745"
    
    return {
        'is_fraud': bool(ensemble_pred),
        'risk_level': risk_level,
        'risk_color': risk_color,
        'ensemble_score': round(float(ensemble_score), 4),
        'xgboost_probability': round(float(xgb_proba), 4),
        'autoencoder_score': round(float(ae_score), 4),
        'models_agree': bool(xgb_pred == ae_pred)
    }


def generate_shap_plot(features_df):
    """Generate SHAP explanation plot."""
    shap_values = explainer.shap_values(features_df)
    
    # Get top 10 features
    feature_impacts = []
    for i, name in enumerate(feature_names):
        feature_impacts.append({
            'feature': name,
            'value': float(features_df.iloc[0][name]),
            'impact': float(shap_values[0][i])
        })
    
    feature_impacts.sort(key=lambda x: abs(x['impact']), reverse=True)
    
    # Create bar chart
    top_features = feature_impacts[:10]
    features_reversed = list(reversed([f['feature'] for f in top_features]))
    impacts_reversed = list(reversed([f['impact'] for f in top_features]))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#dc3545' if x > 0 else '#28a745' for x in impacts_reversed]
    ax.barh(features_reversed, impacts_reversed, color=colors)
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.set_xlabel('SHAP Impact on Fraud Probability')
    ax.set_title('Top Features Influencing This Prediction')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Convert to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    
    return img_base64, feature_impacts[:10]


# ============================================
# ROUTES
# ============================================

@app.route('/')
def home():
    """Home page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Predict fraud for a single transaction from form."""
    try:
        transaction = {
            'step': int(request.form.get('step', 1)),
            'type': request.form.get('type', 'PAYMENT').upper(),
            'amount': float(request.form.get('amount', 0)),
            'oldbalanceOrg': float(request.form.get('oldbalanceOrg', 0)),
            'newbalanceOrig': float(request.form.get('newbalanceOrig', 0)),
            'oldbalanceDest': float(request.form.get('oldbalanceDest', 0)),
            'newbalanceDest': float(request.form.get('newbalanceDest', 0)),
            'hour': int(request.form.get('hour', 12)),
            'day_of_week': int(request.form.get('day_of_week', 3)),
            'age': int(request.form.get('age', 35)),
            'gender': request.form.get('gender', 'M').upper(),
            'txn_count_24h': int(request.form.get('txn_count_24h', 1)),
            'device': request.form.get('device', 'Web').capitalize(),
            'location_distance': float(request.form.get('location_distance', 5)),
            'previous_frauds': int(request.form.get('previous_frauds', 0)),
            'account_age_days': int(request.form.get('account_age_days', 365))
        }
        
        result = predict_single(transaction)
        
        # Generate SHAP explanation
        features_df = preprocess_transaction(transaction)
        shap_img, top_features = generate_shap_plot(features_df)
        
        return render_template('result.html', 
                              result=result,
                              transaction=transaction,
                              shap_img=shap_img,
                              top_features=top_features)
    
    except Exception as e:
        return render_template('error.html', error=str(e))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for single prediction."""
    try:
        data = request.get_json()
        result = predict_single(data)
        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/batch', methods=['POST'])
def api_batch():
    """API endpoint for batch prediction from uploaded CSV."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        df = pd.read_csv(file)
        
        results = []
        for _, row in df.iterrows():
            transaction = row.to_dict()
            result = predict_single(transaction)
            result['transaction_id'] = len(results)
            results.append(result)
        
        results_df = pd.DataFrame(results)
        
        # Save to buffer
        output = io.BytesIO()
        results_df.to_csv(output, index=False)
        output.seek(0)
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name='fraud_detection_results.csv'
        )
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/sample/normal')
def sample_normal():
    """Return a sample normal transaction."""
    return jsonify({
        'step': 480,
        'type': 'PAYMENT',
        'amount': 250.00,
        'oldbalanceOrg': 5000.00,
        'newbalanceOrig': 4750.00,
        'oldbalanceDest': 10000.00,
        'newbalanceDest': 10250.00,
        'hour': 14,
        'day_of_week': 3,
        'age': 35,
        'gender': 'F',
        'txn_count_24h': 3,
        'device': 'Web',
        'location_distance': 5.0,
        'previous_frauds': 0,
        'account_age_days': 500
    })


@app.route('/sample/fraud')
def sample_fraud():
    """Return a sample fraudulent transaction."""
    return jsonify({
        'step': 360,
        'type': 'TRANSFER',
        'amount': 15000.00,
        'oldbalanceOrg': 20000.00,
        'newbalanceOrig': 20000.00,
        'oldbalanceDest': 500.00,
        'newbalanceDest': 500.00,
        'hour': 3,
        'day_of_week': 6,
        'age': 65,
        'gender': 'M',
        'txn_count_24h': 12,
        'device': 'Mobile',
        'location_distance': 500.0,
        'previous_frauds': 2,
        'account_age_days': 15
    })


# ============================================
# MAIN
# ============================================
if __name__ == '__main__':
    load_models()
    print("\n" + "="*60)
    print("🌐 FRAUD DETECTION WEB APP STARTING...")
    print("="*60)
    print("\n📍 Open your browser and go to: http://127.0.0.1:5000")
    print("\n📋 Features:")
    print("   - Single transaction check")
    print("   - Batch CSV upload")
    print("   - SHAP explanations")
    print("   - Risk level indicators")
    print("\n" + "="*60)
    app.run(debug=True, host='127.0.0.1', port=5000)