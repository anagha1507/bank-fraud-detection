import numpy as np
import pandas as pd
import pickle
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve,
                             f1_score, accuracy_score, average_precision_score,
                             roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('default')
sns.set_palette("husl")


class FraudDetectionEnsemble:
    """
    Ensemble fraud detection system combining:
    1. XGBoost (Supervised) - Pattern-based detection
    2. Autoencoder (Unsupervised) - Anomaly-based detection
    """
    
    def __init__(self):
        self.xgboost_model = None
        self.autoencoder_model = None
        self.xgboost_threshold = 0.5
        self.autoencoder_threshold = None
        self.feature_names = None
        self.ensemble_weights = {'xgboost': 0.6, 'autoencoder': 0.4}
        
    def load_models(self, xgboost_path='../models/xgboost_fraud_model.pkl',
                    autoencoder_path='../models/autoencoder_model'):
        """Load both trained models."""
        print("\n" + "="*60)
        print("🔧 LOADING TRAINED MODELS")
        print("="*60)
        
        print("\n📦 Loading XGBoost model...")
        with open(xgboost_path, 'rb') as f:
            xgb_data = pickle.load(f)
        self.xgboost_model = xgb_data['model']
        self.feature_names = xgb_data['feature_names']
        self.xgboost_threshold = xgb_data.get('optimal_threshold', 0.5)
        print(f"✅ XGBoost loaded (threshold: {self.xgboost_threshold:.4f})")
        
        print("\n📦 Loading Autoencoder model...")
        import tensorflow as tf
        self.autoencoder_model = tf.keras.models.load_model(
            f'{autoencoder_path}/autoencoder.keras'
        )
        with open(f'{autoencoder_path}/metadata.pkl', 'rb') as f:
            ae_metadata = pickle.load(f)
        self.autoencoder_threshold = ae_metadata['threshold']
        print(f"✅ Autoencoder loaded (threshold: {self.autoencoder_threshold:.6f})")
        
        print("\n✅ Both models loaded successfully!")
        
    def predict_xgboost(self, X):
        """Get XGBoost predictions."""
        probas = self.xgboost_model.predict_proba(X)[:, 1]
        predictions = (probas >= self.xgboost_threshold).astype(int)
        return probas, predictions
    
    def predict_autoencoder(self, X):
        """Get Autoencoder anomaly scores."""
        X_pred = self.autoencoder_model.predict(X, verbose=0)
        mse = np.mean(np.square(X - X_pred), axis=1)
        anomaly_score = mse / self.autoencoder_threshold
        predictions = (mse > self.autoencoder_threshold).astype(int)
        return mse, anomaly_score, predictions
    
    def ensemble_predict(self, X, weights=None):
        """Combined prediction from both models."""
        if weights is None:
            weights = self.ensemble_weights
        
        xgb_proba, xgb_pred = self.predict_xgboost(X)
        ae_mse, ae_score, ae_pred = self.predict_autoencoder(X)
        
        ae_normalized = 1 / (1 + np.exp(-(ae_score - 1) * 5))
        
        ensemble_score = (weights['xgboost'] * xgb_proba + 
                         weights['autoencoder'] * ae_normalized)
        
        ensemble_threshold = 0.5
        ensemble_pred = (ensemble_score >= ensemble_threshold).astype(int)
        
        return {
            'xgboost_proba': xgb_proba,
            'xgboost_pred': xgb_pred,
            'autoencoder_error': ae_mse,
            'autoencoder_score': ae_score,
            'autoencoder_pred': ae_pred,
            'ensemble_score': ensemble_score,
            'ensemble_pred': ensemble_pred
        }
    
    def evaluate_all_models(self, X_test, y_test):
        """Comprehensive evaluation comparing all three approaches."""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE MODEL EVALUATION")
        print("="*60)
        
        results = self.ensemble_predict(X_test)
        
        models = {
            'XGBoost (Supervised)': (results['xgboost_pred'], results['xgboost_proba']),
            'Autoencoder (Unsupervised)': (results['autoencoder_pred'], results['autoencoder_score']),
            'ENSEMBLE (Combined)': (results['ensemble_pred'], results['ensemble_score'])
        }
        
        comparison_results = {}
        
        print("\n" + "="*70)
        print(f"{'MODEL':<30} {'ACC':>8} {'F1':>8} {'ROC-AUC':>8} {'PR-AUC':>8}")
        print("="*70)
        
        for model_name, (y_pred, y_score) in models.items():
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc = roc_auc_score(y_test, y_score)
            pr = average_precision_score(y_test, y_score)
            
            comparison_results[model_name] = {
                'accuracy': acc, 'f1_score': f1, 'roc_auc': roc, 'pr_auc': pr
            }
            
            print(f"{model_name:<30} {acc:>8.4f} {f1:>8.4f} {roc:>8.4f} {pr:>8.4f}")
        
        print("="*70)
        
        print(f"\n{'='*60}")
        print("🎯 ENSEMBLE MODEL - DETAILED RESULTS")
        print("="*60)
        
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, results['ensemble_pred'])
        print(f"  ✅ True Negatives: {cm[0,0]:,}")
        print(f"  ⚠️  False Positives: {cm[0,1]:,}")
        print(f"  🚨 False Negatives (Missed): {cm[1,0]:,}")
        print(f"  ✅ True Positives (Caught): {cm[1,1]:,}")
        
        fn_cost = cm[1,0] * 10
        fp_cost = cm[0,1] * 1
        total_cost = fn_cost + fp_cost
        print(f"\n💰 Cost Analysis:")
        print(f"  False Negative Cost: {fn_cost:,} units")
        print(f"  False Positive Cost: {fp_cost:,} units")
        print(f"  Total Cost: {total_cost:,} units")
        
        print(f"\n🤝 Model Agreement Analysis:")
        agree = (results['xgboost_pred'] == results['autoencoder_pred']).sum()
        disagree = len(y_test) - agree
        print(f"  Models agree on: {agree:,} transactions ({agree/len(y_test)*100:.1f}%)")
        print(f"  Models disagree on: {disagree:,} transactions ({disagree/len(y_test)*100:.1f}%)")
        
        agree_indices = results['xgboost_pred'] == results['autoencoder_pred']
        agree_accuracy = (results['xgboost_pred'][agree_indices] == y_test[agree_indices]).mean()
        print(f"  Accuracy when models agree: {agree_accuracy:.4f}")
        
        return comparison_results, results
    
    def plot_roc_comparison(self, X_test, y_test, save_path='../models/roc_comparison.png'):
        """Plot ROC curves for all models."""
        print("\n📈 Generating ROC comparison...")
        
        results = self.ensemble_predict(X_test)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        models_data = [
            ('XGBoost', results['xgboost_proba'], '#4ECDC4'),
            ('Autoencoder', results['autoencoder_score'], '#FF6B6B'),
            ('ENSEMBLE', results['ensemble_score'], '#45B7D1')
        ]
        
        for name, scores, color in models_data:
            fpr, tpr, _ = roc_curve(y_test, scores)
            auc = roc_auc_score(y_test, scores)
            ax.plot(fpr, tpr, color=color, linewidth=2.5, 
                   label=f'{name} (AUC = {auc:.4f})')
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random (AUC = 0.5)')
        
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title('ROC Curves: Model Comparison', fontsize=15, fontweight='bold')
        ax.legend(loc='lower right', fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ ROC comparison saved to {save_path}")
    
    def plot_score_distributions(self, X_test, y_test, save_path='../models/score_distributions.png'):
        """Plot score distributions for normal vs fraud transactions."""
        print("\n📊 Generating score distribution plots...")
        
        results = self.ensemble_predict(X_test)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        score_data = [
            ('XGBoost Probability', results['xgboost_proba'], axes[0], '#4ECDC4'),
            ('Autoencoder Score', results['autoencoder_score'], axes[1], '#FF6B6B'),
            ('Ensemble Score', results['ensemble_score'], axes[2], '#45B7D1')
        ]
        
        for title, scores, ax, color in score_data:
            normal_scores = scores[y_test == 0]
            fraud_scores = scores[y_test == 1]
            
            ax.hist(normal_scores, bins=30, alpha=0.6, label='Normal', 
                   color='#4ECDC4', edgecolor='white')
            ax.hist(fraud_scores, bins=30, alpha=0.6, label='Fraud', 
                   color='#FF6B6B', edgecolor='white')
            
            ax.set_xlabel('Score', fontsize=10)
            ax.set_ylabel('Count', fontsize=10)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Score Distributions: Normal vs Fraud Transactions', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Score distributions saved to {save_path}")
    
    def analyze_disagreements(self, X_test, y_test, n_examples=5):
        """Analyze cases where XGBoost and Autoencoder disagree."""
        print("\n" + "="*60)
        print("🔍 ANALYZING MODEL DISAGREEMENTS")
        print("="*60)
        
        results = self.ensemble_predict(X_test)
        
        # Convert to numpy arrays for position-based indexing
        y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
        xgb_pred_np = np.array(results['xgboost_pred'])
        ae_pred_np = np.array(results['autoencoder_pred'])
        xgb_proba_np = np.array(results['xgboost_proba'])
        ae_score_np = np.array(results['autoencoder_score'])
        
        # Case 1: XGBoost says FRAUD, Autoencoder says NORMAL
        xgb_fraud_ae_normal = (xgb_pred_np == 1) & (ae_pred_np == 0)
        
        print(f"\n📊 XGBoost=FRAUD, Autoencoder=NORMAL: {xgb_fraud_ae_normal.sum()} cases")
        if xgb_fraud_ae_normal.sum() > 0:
            disagree_indices = np.where(xgb_fraud_ae_normal)[0][:n_examples]
            print("\nSample cases (XGBoost catches what Autoencoder misses):")
            for i, pos in enumerate(disagree_indices):
                actual = "FRAUD" if y_test_np[pos] == 1 else "NORMAL"
                print(f"  Case {i+1}: XGBoost prob={xgb_proba_np[pos]:.4f}, "
                      f"AE score={ae_score_np[pos]:.4f}, "
                      f"Actual={actual}")
        
        # Case 2: Autoencoder says FRAUD, XGBoost says NORMAL
        ae_fraud_xgb_normal = (ae_pred_np == 1) & (xgb_pred_np == 0)
        
        print(f"\n📊 Autoencoder=FRAUD, XGBoost=NORMAL: {ae_fraud_xgb_normal.sum()} cases")
        if ae_fraud_xgb_normal.sum() > 0:
            disagree_indices = np.where(ae_fraud_xgb_normal)[0][:n_examples]
            print("\nSample cases (Autoencoder catches novel patterns XGBoost misses):")
            for i, pos in enumerate(disagree_indices):
                actual = "FRAUD" if y_test_np[pos] == 1 else "NORMAL"
                print(f"  Case {i+1}: XGBoost prob={xgb_proba_np[pos]:.4f}, "
                      f"AE score={ae_score_np[pos]:.4f}, "
                      f"Actual={actual}")
        
        return results
    
    def generate_final_report(self, X_test, y_test):
        """Generate comprehensive final evaluation report."""
        print("\n" + "="*60)
        print("📋 GENERATING FINAL EVALUATION REPORT")
        print("="*60)
        
        results = self.ensemble_predict(X_test)
        
        report = []
        report.append("="*80)
        report.append("BANK FRAUD DETECTION SYSTEM - FINAL EVALUATION REPORT")
        report.append("="*80)
        report.append("")
        report.append(f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("1. SYSTEM ARCHITECTURE")
        report.append("-" * 40)
        report.append("Components:")
        report.append("  * XGBoost Classifier (Supervised Learning)")
        report.append("    - Cost-sensitive training (10:1 penalty for missed fraud)")
        report.append("    - SHAP explainability for each prediction")
        report.append("  * Autoencoder (Unsupervised Learning)")
        report.append("    - Trained on normal transactions only")
        report.append("    - Detects anomalies via reconstruction error")
        report.append("  * Ensemble System")
        report.append("    - Weighted combination of both models")
        report.append("    - Captures both known and novel fraud patterns")
        report.append("")
        
        report.append("2. PERFORMANCE SUMMARY")
        report.append("-" * 40)
        
        models = {
            'XGBoost': (results['xgboost_pred'], results['xgboost_proba']),
            'Autoencoder': (results['autoencoder_pred'], results['autoencoder_score']),
            'Ensemble': (results['ensemble_pred'], results['ensemble_score'])
        }
        
        for name, (pred, score) in models.items():
            report.append(f"\n{name}:")
            report.append(f"  Accuracy: {accuracy_score(y_test, pred):.4f}")
            report.append(f"  F1-Score: {f1_score(y_test, pred):.4f}")
            report.append(f"  ROC-AUC: {roc_auc_score(y_test, score):.4f}")
            report.append(f"  PR-AUC: {average_precision_score(y_test, score):.4f}")
            
            cm = confusion_matrix(y_test, pred)
            report.append(f"  True Positives: {cm[1,1]:,}")
            report.append(f"  False Positives: {cm[0,1]:,}")
            report.append(f"  False Negatives: {cm[1,0]:,}")
            report.append(f"  True Negatives: {cm[0,0]:,}")
        
        report.append("")
        
        report.append("3. BUSINESS IMPACT")
        report.append("-" * 40)
        
        ensemble_cm = confusion_matrix(y_test, results['ensemble_pred'])
        fn_cost = ensemble_cm[1,0] * 10
        fp_cost = ensemble_cm[0,1] * 1
        total_cost = fn_cost + fp_cost
        
        report.append(f"  Total Transactions Analyzed: {len(y_test):,}")
        report.append(f"  Fraud Transactions Caught: {ensemble_cm[1,1]:,}")
        report.append(f"  Fraud Transactions Missed: {ensemble_cm[1,0]:,}")
        report.append(f"  False Alarms: {ensemble_cm[0,1]:,}")
        report.append(f"  Total Cost of Errors: {total_cost:,} units")
        report.append("")
        
        report.append("4. RECOMMENDATIONS")
        report.append("-" * 40)
        report.append("  * Deploy ensemble model for production fraud detection")
        report.append("  * Use SHAP explanations in fraud analyst review workflow")
        report.append("  * Monitor autoencoder reconstruction error for new fraud patterns")
        report.append("  * Retrain models monthly with new transaction data")
        report.append("  * Set up alerts for high ensemble scores (>0.8)")
        report.append("")
        
        report.append("="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        report_text = "\n".join(report)
        report_path = '../models/final_evaluation_report.txt'
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n✅ Final report saved to {report_path}")
        
        return report_text
    
    def save_ensemble_config(self, path='../models/ensemble_config.json'):
        """Save ensemble configuration."""
        config = {
            'xgboost_threshold': float(self.xgboost_threshold),
            'autoencoder_threshold': float(self.autoencoder_threshold),
            'ensemble_weights': self.ensemble_weights,
            'feature_names': self.feature_names
        }
        
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Ensemble config saved to {path}")


# Main execution
if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from data_preprocessing import FraudDataPreprocessor
    
    print("\n" + "="*60)
    print("🏆 FINAL EVALUATION: COMPLETE FRAUD DETECTION SYSTEM")
    print("="*60)
    
    # Step 1: Prepare data
    print("\n🔹 Step 1: Preparing test data...")
    preprocessor = FraudDataPreprocessor(n_samples=50000, fraud_ratio=0.02)
    df = preprocessor.generate_synthetic_data()
    X_train, X_test, y_train, y_test, num_cols, cat_cols = preprocessor.preprocess(df)
    
    # Step 2: Initialize ensemble
    print("\n🔹 Step 2: Initializing ensemble system...")
    ensemble = FraudDetectionEnsemble()
    
    # Step 3: Load models
    print("\n🔹 Step 3: Loading models...")
    
    xgb_path = '../models/xgboost_fraud_model.pkl'
    ae_path = '../models/autoencoder_model'
    
    if not os.path.exists(xgb_path):
        print("XGBoost model not found. Training now...")
        from xgboost_model import FraudDetectionXGBoost
        xgb_detector = FraudDetectionXGBoost()
        xgb_detector.train_with_cost_sensitive(
            X_train, y_train, X_test, y_test,
            feature_names=X_train.columns.tolist(),
            cost_ratio=10
        )
        xgb_detector.save_model(xgb_path)
    
    if not os.path.exists(f'{ae_path}/autoencoder.keras'):
        print("Autoencoder not found. Training now...")
        from autoencoder_model import FraudAutoencoder
        X_train_normal = X_train[y_train == 0]
        autoencoder = FraudAutoencoder(input_dim=X_train.shape[1], encoding_dim=16)
        autoencoder.train(X_train_normal, epochs=30, batch_size=256, 
                         early_stopping_patience=5, verbose=0)
        autoencoder.find_optimal_threshold(X_train_normal, X_test, y_test)
        autoencoder.save_model(ae_path)
    
    ensemble.load_models(xgb_path, ae_path)
    
    # Step 4: Comprehensive evaluation
    print("\n🔹 Step 4: Running comprehensive evaluation...")
    comparison_results, predictions = ensemble.evaluate_all_models(X_test, y_test)
    
    # Step 5: Visualizations
    print("\n🔹 Step 5: Generating comparison visualizations...")
    ensemble.plot_roc_comparison(X_test, y_test)
    ensemble.plot_score_distributions(X_test, y_test)
    
    # Step 6: Analyze disagreements
    print("\n🔹 Step 6: Analyzing model disagreements...")
    ensemble.analyze_disagreements(X_test, y_test)
    
    # Step 7: Generate final report
    print("\n🔹 Step 7: Generating final report...")
    ensemble.generate_final_report(X_test, y_test)
    
    # Step 8: Save configuration
    print("\n🔹 Step 8: Saving ensemble configuration...")
    ensemble.save_ensemble_config()
    
    # Final Summary
    print("\n" + "="*70)
    print("🎉 BANK FRAUD DETECTION SYSTEM - COMPLETE!")
    print("="*70)
    
    print("\n🧠 CORE AI TECHNOLOGIES USED:")
    print("  ✅ XGBoost - Supervised fraud classification")
    print("  ✅ SHAP - Model explainability & interpretability")
    print("  ✅ Autoencoders - Unsupervised anomaly detection")
    print("  ✅ Cost-Sensitive Learning - Optimized for fraud detection")
    
    print("\n📊 KEY CAPABILITIES:")
    print("  ✅ Synthetic data generation with realistic fraud patterns")
    print("  ✅ Feature engineering (risk scores, time patterns, ratios)")
    print("  ✅ Cost-sensitive XGBoost (10:1 penalty for missed fraud)")
    print("  ✅ SHAP explanations for every prediction")
    print("  ✅ Autoencoder for novel fraud pattern detection")
    print("  ✅ Ensemble model combining both approaches")
    print("  ✅ Comprehensive evaluation & reporting")
    
    print("\n" + "="*70)
    print("✅ ALL DONE! Your fraud detection system is ready!")
    print("="*70)