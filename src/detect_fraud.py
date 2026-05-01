"""
BANK FRAUD DETECTION - PREDICTION SCRIPT
Use this to detect fraud in new transactions
"""
import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf


class FraudDetector:
    """
    Production-ready fraud detection system.
    Uses both XGBoost and Autoencoder for robust detection.
    """
    
    def __init__(self):
        """Load all trained models and configurations."""
        print("🔧 Loading fraud detection models...")
        
        # Load XGBoost
        with open('../models/xgboost_fraud_model.pkl', 'rb') as f:
            xgb_data = pickle.load(f)
        self.xgboost_model = xgb_data['model']
        self.xgboost_threshold = xgb_data['optimal_threshold']
        self.feature_names = xgb_data['feature_names']
        
        # Load Autoencoder
        self.autoencoder_model = tf.keras.models.load_model(
            '../models/autoencoder_model/autoencoder.keras'
        )
        with open('../models/autoencoder_model/metadata.pkl', 'rb') as f:
            ae_metadata = pickle.load(f)
        self.ae_threshold = ae_metadata['threshold']
        
        print("✅ Models loaded successfully!")
        print(f"   XGBoost threshold: {self.xgboost_threshold:.4f}")
        print(f"   Autoencoder threshold: {self.ae_threshold:.6f}")
        print(f"   Features required: {len(self.feature_names)}")
        
    def preprocess_transaction(self, transaction_dict):
        """
        Preprocess a single transaction for prediction.
        
        Input: Dictionary with transaction details
        Output: Processed feature vector matching training format
        """
        # Create dataframe from transaction
        df = pd.DataFrame([transaction_dict])
        
        # ============================================
        # FEATURE ENGINEERING (Same as training)
        # ============================================
        
        # Balance changes
        df['balance_change_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        
        # Error balances
        df['errorBalanceOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
        df['errorBalanceDest'] = df['oldbalanceDest'] - df['newbalanceDest']
        
        # Ratio features
        df['amount_to_old_balance'] = df['amount'] / (df['oldbalanceOrg'] + 1)
        df['amount_to_new_balance'] = df['amount'] / (df['newbalanceOrig'] + 1)
        df['balance_ratio'] = df['oldbalanceOrg'] / (df['oldbalanceDest'] + 1)
        
        # Time features
        df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Amount features
        df['amount_log'] = np.log1p(df['amount'])
        df['high_amount'] = (df['amount'] > 10000).astype(int)  # Threshold can be adjusted
        
        # Risk score (same formula as training)
        df['risk_score'] = (
            (df['high_amount'] * 0.3) +
            (df['is_night'] * 0.1) +
            (df['is_weekend'] * 0.05) +
            ((df['txn_count_24h'] > 10).astype(int) * 0.2) +
            ((df['location_distance'] > 100).astype(int) * 0.2) +
            ((df['previous_frauds'] > 0).astype(int) * 0.15)
        )
        
        # ============================================
        # ENCODING CATEGORICALS (Same as training)
        # ============================================
        
        # Type encoding
        type_map = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4}
        df['type'] = df['type'].map(type_map).fillna(0)
        
        # Gender encoding
        gender_map = {'M': 0, 'F': 1}
        df['gender'] = df['gender'].map(gender_map).fillna(0)
        
        # Device encoding
        device_map = {'Mobile': 0, 'Web': 1, 'ATM': 2, 'POS': 3}
        df['device'] = df['device'].map(device_map).fillna(0)
        
        # ============================================
        # ENSURE CORRECT FEATURE ORDER
        # ============================================
        
        # Add missing columns with 0
        for col in self.feature_names:
            if col not in df.columns:
                df[col] = 0
        
        # Keep only required columns in correct order
        df = df[self.feature_names]
        
        return df
    
    def detect_fraud(self, transaction_dict):
        """
        Detect fraud for a single transaction.
        
        Args:
            transaction_dict: Dictionary with transaction details
            
        Returns:
            Dictionary with fraud detection results
        """
        # Preprocess
        features = self.preprocess_transaction(transaction_dict)
        
        # ============================================
        # XGBOOST PREDICTION
        # ============================================
        xgb_proba = self.xgboost_model.predict_proba(features)[0, 1]
        xgb_prediction = xgb_proba >= self.xgboost_threshold
        
        # ============================================
        # AUTOENCODER PREDICTION
        # ============================================
        features_array = features.values
        reconstructed = self.autoencoder_model.predict(features_array, verbose=0)
        mse = np.mean(np.square(features_array - reconstructed), axis=1)[0]
        ae_score = mse / self.ae_threshold
        ae_prediction = mse > self.ae_threshold
        
        # ============================================
        # ENSEMBLE DECISION
        # ============================================
        ae_normalized = 1 / (1 + np.exp(-(ae_score - 1) * 5))
        ensemble_score = (0.6 * xgb_proba) + (0.4 * ae_normalized)
        ensemble_prediction = ensemble_score >= 0.5
        
        # ============================================
        # RISK LEVEL
        # ============================================
        if ensemble_score >= 0.8:
            risk_level = "🔴 CRITICAL"
        elif ensemble_score >= 0.6:
            risk_level = "🟠 HIGH"
        elif ensemble_score >= 0.4:
            risk_level = "🟡 MEDIUM"
        else:
            risk_level = "🟢 LOW"
        
        # ============================================
        # RESULT
        # ============================================
        result = {
            'is_fraud': bool(ensemble_prediction),
            'risk_level': risk_level,
            'ensemble_score': round(float(ensemble_score), 4),
            'xgboost': {
                'fraud_probability': round(float(xgb_proba), 4),
                'is_fraud': bool(xgb_prediction)
            },
            'autoencoder': {
                'anomaly_score': round(float(ae_score), 4),
                'reconstruction_error': round(float(mse), 6),
                'is_anomaly': bool(ae_prediction)
            },
            'models_agree': bool(xgb_prediction == ae_prediction)
        }
        
        return result
    
    def detect_fraud_batch(self, transactions_df):
        """
        Detect fraud for multiple transactions.
        
        Args:
            transactions_df: DataFrame with transaction details
            
        Returns:
            DataFrame with fraud detection results
        """
        results = []
        
        for idx, row in transactions_df.iterrows():
            transaction = row.to_dict()
            result = self.detect_fraud(transaction)
            result['transaction_id'] = idx
            results.append(result)
        
        results_df = pd.DataFrame(results)
        return results_df
    
    def explain_prediction(self, transaction_dict):
        """
        Provide SHAP explanation for why a transaction was flagged.
        """
        import shap
        
        features = self.preprocess_transaction(transaction_dict)
        
        # Create explainer
        explainer = shap.TreeExplainer(self.xgboost_model)
        shap_values = explainer.shap_values(features)
        
        # Get top contributing features
        contributions = []
        for i, feature_name in enumerate(self.feature_names):
            contributions.append({
                'feature': feature_name,
                'value': float(features.iloc[0][feature_name]),
                'impact': float(shap_values[0][i]),
                'direction': 'Increases' if shap_values[0][i] > 0 else 'Decreases'
            })
        
        # Sort by absolute impact
        contributions.sort(key=lambda x: abs(x['impact']), reverse=True)
        
        return contributions[:10]  # Top 10 features


# ============================================
# INTERACTIVE DEMO
# ============================================
def generate_sample_transaction(is_fraud=False):
    """Generate a sample transaction for testing."""
    
    if is_fraud:
        # Fraudulent transaction pattern
        return {
            'step': 360,
            'type': 'TRANSFER',
            'amount': 15000.00,
            'oldbalanceOrg': 20000.00,
            'newbalanceOrig': 20000.00,  # Balance didn't change (suspicious)
            'oldbalanceDest': 500.00,
            'newbalanceDest': 500.00,    # Destination balance unchanged
            'hour': 3,                    # Night time
            'day_of_week': 6,            # Weekend
            'age': 65,
            'gender': 'M',
            'txn_count_24h': 12,         # Unusual activity
            'device': 'Mobile',
            'location_distance': 500.0,  # Far from home
            'previous_frauds': 2,
            'account_age_days': 15       # New account
        }
    else:
        # Normal transaction pattern
        return {
            'step': 480,
            'type': 'PAYMENT',
            'amount': 250.00,
            'oldbalanceOrg': 5000.00,
            'newbalanceOrig': 4750.00,
            'oldbalanceDest': 10000.00,
            'newbalanceDest': 10250.00,
            'hour': 14,                   # Daytime
            'day_of_week': 3,            # Weekday
            'age': 35,
            'gender': 'F',
            'txn_count_24h': 3,
            'device': 'Web',
            'location_distance': 5.0,
            'previous_frauds': 0,
            'account_age_days': 500      # Old account
        }


def interactive_demo():
    """Run an interactive fraud detection demo."""
    
    print("\n" + "="*60)
    print("🏦 BANK FRAUD DETECTION SYSTEM")
    print("="*60)
    
    # Initialize detector
    detector = FraudDetector()
    
    while True:
        print("\n" + "-"*60)
        print("OPTIONS:")
        print("  1. Test with a NORMAL transaction")
        print("  2. Test with a FRAUDULENT transaction")
        print("  3. Enter custom transaction details")
        print("  4. Batch detection from file")
        print("  5. Exit")
        print("-"*60)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            # Normal transaction
            print("\n📊 Testing NORMAL transaction...")
            transaction = generate_sample_transaction(is_fraud=False)
            
            print("\nTransaction Details:")
            for key, value in transaction.items():
                print(f"  {key}: {value}")
            
            result = detector.detect_fraud(transaction)
            
            print("\n🔍 Detection Result:")
            print(f"  Risk Level: {result['risk_level']}")
            print(f"  Ensemble Score: {result['ensemble_score']}")
            print(f"  Is Fraud: {'🚨 YES' if result['is_fraud'] else '✅ NO'}")
            print(f"  XGBoost Probability: {result['xgboost']['fraud_probability']}")
            print(f"  Autoencoder Score: {result['autoencoder']['anomaly_score']}")
            print(f"  Models Agree: {'Yes' if result['models_agree'] else 'No'}")
            
            if result['is_fraud']:
                print("\n⚠️  FLAGGED AS FRAUD! Getting explanation...")
                explanations = detector.explain_prediction(transaction)
                print("\nTop reasons for fraud flag:")
                for exp in explanations[:5]:
                    direction = "🔺" if exp['impact'] > 0 else "🔻"
                    print(f"  {direction} {exp['feature']}: impact={exp['impact']:.4f}")
        
        elif choice == '2':
            # Fraudulent transaction
            print("\n📊 Testing FRAUDULENT transaction...")
            transaction = generate_sample_transaction(is_fraud=True)
            
            print("\nTransaction Details:")
            for key, value in transaction.items():
                print(f"  {key}: {value}")
            
            result = detector.detect_fraud(transaction)
            
            print("\n🔍 Detection Result:")
            print(f"  Risk Level: {result['risk_level']}")
            print(f"  Ensemble Score: {result['ensemble_score']}")
            print(f"  Is Fraud: {'🚨 YES' if result['is_fraud'] else '✅ NO'}")
            print(f"  XGBoost Probability: {result['xgboost']['fraud_probability']}")
            print(f"  Autoencoder Score: {result['autoencoder']['anomaly_score']}")
            print(f"  Models Agree: {'Yes' if result['models_agree'] else 'No'}")
            
            if result['is_fraud']:
                print("\n⚠️  FLAGGED AS FRAUD! Getting explanation...")
                explanations = detector.explain_prediction(transaction)
                print("\nTop reasons for fraud flag:")
                for exp in explanations[:5]:
                    direction = "🔺" if exp['impact'] > 0 else "🔻"
                    print(f"  {direction} {exp['feature']}: impact={exp['impact']:.4f}")
        
        elif choice == '3':
            # Custom transaction
            print("\n📝 Enter custom transaction details:")
            
            transaction = {}
            transaction['amount'] = float(input("  Amount ($): "))
            transaction['type'] = input("  Type (CASH_IN/CASH_OUT/DEBIT/PAYMENT/TRANSFER): ").upper()
            transaction['oldbalanceOrg'] = float(input("  Origin account balance before ($): "))
            transaction['newbalanceOrig'] = float(input("  Origin account balance after ($): "))
            transaction['oldbalanceDest'] = float(input("  Destination balance before ($): "))
            transaction['newbalanceDest'] = float(input("  Destination balance after ($): "))
            transaction['hour'] = int(input("  Hour of day (0-23): "))
            transaction['day_of_week'] = int(input("  Day of week (0=Mon, 6=Sun): "))
            transaction['age'] = int(input("  Customer age: "))
            transaction['gender'] = input("  Gender (M/F): ").upper()
            transaction['txn_count_24h'] = int(input("  Transactions in last 24h: "))
            transaction['device'] = input("  Device (Mobile/Web/ATM/POS): ").capitalize()
            transaction['location_distance'] = float(input("  Distance from home branch (km): "))
            transaction['previous_frauds'] = int(input("  Previous fraud count: "))
            transaction['account_age_days'] = int(input("  Account age (days): "))
            transaction['step'] = 1
            
            result = detector.detect_fraud(transaction)
            
            print(f"\n🔍 Result: {result['risk_level']}")
            print(f"  Ensemble Score: {result['ensemble_score']}")
            print(f"  Is Fraud: {'🚨 YES' if result['is_fraud'] else '✅ NO'}")
        
        elif choice == '4':
            # Batch detection
            filepath = input("\n📂 Enter CSV file path: ").strip()
            try:
                df = pd.read_csv(filepath)
                print(f"Loaded {len(df)} transactions...")
                
                results_df = detector.detect_fraud_batch(df)
                
                # Save results
                output_path = '../data/detection_results.csv'
                results_df.to_csv(output_path, index=False)
                
                fraud_count = results_df['is_fraud'].sum()
                print(f"\n✅ Results saved to {output_path}")
                print(f"   Total transactions: {len(results_df)}")
                print(f"   Fraud detected: {fraud_count}")
                print(f"   Normal: {len(results_df) - fraud_count}")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        elif choice == '5':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("\n❌ Invalid choice. Please enter 1-5.")


# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    interactive_demo()