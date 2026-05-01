import numpy as np
import pandas as pd
import pickle
import os
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_auc_score, precision_recall_curve, 
                             f1_score, accuracy_score, average_precision_score)
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionXGBoost:
    """
    XGBoost model for fraud detection with cost-sensitive learning.
    Uses scale_pos_weight to handle class imbalance by penalizing 
    false negatives (missed fraud) more than false positives.
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.best_params = None
        self.optimal_threshold = 0.5
        
    def train_with_cost_sensitive(self, X_train, y_train, X_test, y_test, 
                                   feature_names=None, cost_ratio=10):
        """
        Train XGBoost with cost-sensitive learning.
        
        cost_ratio: How many times more expensive is missing a fraud vs false alarm.
        Default 10 means missing fraud is 10x worse than false positive.
        """
        print("\n" + "="*60)
        print("TRAINING XGBOOST WITH COST-SENSITIVE LEARNING")
        print("="*60)
        
        self.feature_names = feature_names if feature_names is not None else X_train.columns.tolist()
        
        # Calculate class weights
        n_negative = len(y_train[y_train == 0])
        n_positive = len(y_train[y_train == 1])
        
        # scale_pos_weight = (number of negative instances / number of positive instances) * cost_ratio
        scale_pos_weight = (n_negative / n_positive) * cost_ratio
        
        print(f"\nClass Distribution:")
        print(f"  Normal transactions: {n_negative:,}")
        print(f"  Fraudulent transactions: {n_positive:,}")
        print(f"  Fraud ratio: {n_positive/(n_negative+n_positive)*100:.2f}%")
        print(f"  scale_pos_weight: {scale_pos_weight:.2f}")
        print(f"  Cost ratio (FN cost / FP cost): {cost_ratio}:1")
        
        # XGBoost parameters optimized for fraud detection
        params = {
            'objective': 'binary:logistic',
            'eval_metric': ['logloss', 'aucpr'],
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': scale_pos_weight,
            'random_state': self.random_state,
            'n_jobs': -1,
            'tree_method': 'hist',
            'early_stopping_rounds': 30
        }
        
        print("\nTraining model with early stopping...")
        
        # Create evaluation set
        eval_set = [(X_train, y_train), (X_test, y_test)]
        
        # Train model
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=50
        )
        
        print("\n✅ Model training complete!")
        
        # Store best parameters
        self.best_params = self.model.get_params()
        
        # Find and store optimal threshold
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        return self.model
    
    def hyperparameter_tuning(self, X_train, y_train, X_test, y_test, cost_ratio=10):
        """
        Perform grid search for hyperparameter tuning.
        This is optional but can improve performance.
        """
        print("\n" + "="*60)
        print("HYPERPARAMETER TUNING")
        print("="*60)
        
        n_negative = len(y_train[y_train == 0])
        n_positive = len(y_train[y_train == 1])
        scale_pos_weight = (n_negative / n_positive) * cost_ratio
        
        # Parameter grid
        param_grid = {
            'max_depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [200, 300],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.7, 0.8, 0.9],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2]
        }
        
        from sklearn.model_selection import RandomizedSearchCV
        
        # Base model with scale_pos_weight
        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric=['logloss', 'aucpr'],
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            tree_method='hist',
            n_jobs=-1,
            early_stopping_rounds=20
        )
        
        # Randomized search (faster than grid search)
        random_search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=20,
            scoring='average_precision',
            cv=3,
            verbose=1,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        print("Searching for best parameters...")
        random_search.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        print(f"\nBest parameters found:")
        for param, value in random_search.best_params_.items():
            print(f"  {param}: {value}")
        print(f"Best CV score: {random_search.best_score_:.4f}")
        
        # Use best model
        self.model = random_search.best_estimator_
        self.best_params = random_search.best_params_
        
        # Update optimal threshold
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        self.optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        return self.model
    
    def evaluate(self, X_test, y_test, threshold=None):
        """Comprehensive model evaluation with automatic optimal threshold."""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Use optimal threshold if none specified
        if threshold is None:
            threshold = self.optimal_threshold
            print(f"\n📊 Using Optimal Threshold: {threshold:.4f}")
        
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        avg_precision = average_precision_score(y_test, y_pred_proba)
        
        print(f"\n🎯 Performance Metrics (threshold={threshold:.4f}):")
        print(f"  ✅ Accuracy: {accuracy:.4f}")
        print(f"  ✅ F1-Score: {f1:.4f}")
        print(f"  ✅ ROC-AUC: {roc_auc:.4f}")
        print(f"  ✅ PR-AUC (Average Precision): {avg_precision:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📊 Confusion Matrix:")
        print(f"  ✅ True Negatives (Correctly Flagged Normal): {cm[0,0]:,}")
        print(f"  ⚠️  False Positives (False Alarms): {cm[0,1]:,}")
        print(f"  🚨 False Negatives (Missed Fraud!): {cm[1,0]:,}")
        print(f"  ✅ True Positives (Caught Fraud): {cm[1,1]:,}")
        
        # Cost analysis
        fn_cost = cm[1,0] * 10  # Each missed fraud costs 10 units
        fp_cost = cm[0,1] * 1   # Each false alarm costs 1 unit
        total_cost = fn_cost + fp_cost
        print(f"\n💰 Cost Analysis (FN=10x cost, FP=1x cost):")
        print(f"  🚨 False Negative Cost: {fn_cost:,} units")
        print(f"  ⚠️  False Positive Cost: {fp_cost:,} units")
        print(f"  💰 Total Cost: {total_cost:,} units")
        
        # Classification Report
        print(f"\n📋 Detailed Classification Report:")
        print(classification_report(y_test, y_pred, 
                                    target_names=['Normal', 'Fraud']))
        
        # Find optimal threshold
        self._find_optimal_threshold(y_test, y_pred_proba)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'confusion_matrix': cm,
            'y_pred_proba': y_pred_proba,
            'y_pred': y_pred
        }
    
    def _find_optimal_threshold(self, y_test, y_pred_proba):
        """Find the threshold that maximizes F1 score."""
        print("\n" + "-"*40)
        print("🎯 OPTIMAL THRESHOLD ANALYSIS")
        print("-"*40)
        
        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Find threshold that maximizes F1
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        print(f"📊 Optimal threshold (max F1): {optimal_threshold:.4f}")
        print(f"   Precision: {precision[optimal_idx]:.4f}")
        print(f"   Recall: {recall[optimal_idx]:.4f}")
        print(f"   F1-Score: {f1_scores[optimal_idx]:.4f}")
        
        # Find threshold for 90% recall
        target_recall = 0.90
        recall_array = np.array(recall)
        idx_90 = np.argmin(np.abs(recall_array - target_recall))
        threshold_90 = thresholds[idx_90] if idx_90 < len(thresholds) else 0.5
        
        print(f"\n🎯 Threshold for 90% Recall: {threshold_90:.4f}")
        print(f"   Precision: {precision[idx_90]:.4f}")
        print(f"   Recall: {recall[idx_90]:.4f}")
        
        # Find threshold for 95% recall
        target_recall = 0.95
        idx_95 = np.argmin(np.abs(recall_array - target_recall))
        threshold_95 = thresholds[idx_95] if idx_95 < len(thresholds) else 0.5
        
        print(f"\n🎯 Threshold for 95% Recall: {threshold_95:.4f}")
        print(f"   Precision: {precision[idx_95]:.4f}")
        print(f"   Recall: {recall[idx_95]:.4f}")
        
        return optimal_threshold, threshold_90
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance."""
        print("\n" + "="*60)
        print("📊 FEATURE IMPORTANCE ANALYSIS")
        print("="*60)
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features:")
        for i, (idx, row) in enumerate(importance_df.head(top_n).iterrows()):
            print(f"  {i+1}. {row['feature']:<25} {row['importance']:.4f}")
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        top_features = importance_df.head(top_n)
        
        # Highlight key fraud indicators
        key_features = ['amount', 'errorBalanceOrig', 'risk_score', 
                       'txn_count_24h', 'location_distance', 'amount_log',
                       'balance_change_orig', 'high_amount']
        
        colors = ['#FF6B6B' if feat in key_features else '#4ECDC4' 
                  for feat in top_features['feature']]
        
        bars = ax.barh(range(len(top_features)), top_features['importance'].values, 
                      color=colors, edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'].values, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score', fontsize=12)
        ax.set_title(f'Top {top_n} Features for Fraud Detection\n🔴 = Key Fraud Indicators', 
                    fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, top_features['importance'].values)):
            ax.text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                   f'{val:.3f}', va='center', fontsize=9)
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        
        # Ensure models directory exists
        os.makedirs('../models', exist_ok=True)
        plt.savefig('../models/feature_importance_xgboost.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print insight
        print(f"\n💡 Key Insight: Top 3 features account for {importance_df.head(3)['importance'].sum()*100:.1f}% of predictions")
        
        return importance_df
    
    def plot_training_history(self):
        """Plot training and validation metrics (fixed version)."""
        print("\n" + "="*60)
        print("📈 TRAINING HISTORY")
        print("="*60)
        
        if not hasattr(self.model, 'evals_result'):
            print("No training history available.")
            return
        
        results = self.model.evals_result()
        
        if 'validation_0' not in results or 'validation_1' not in results:
            print("No validation results found.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Get available metrics
        available_metrics = list(results['validation_0'].keys())
        print(f"Available metrics: {available_metrics}")
        
        # Plot 1: Find available loss metric
        loss_metric = None
        for metric in ['logloss', 'log_loss', 'error']:
            if metric in available_metrics:
                loss_metric = metric
                break
        
        if loss_metric:
            epochs = len(results['validation_0'][loss_metric])
            x_axis = range(0, epochs)
            
            axes[0].plot(x_axis, results['validation_0'][loss_metric], 
                       label='Training', color='#4ECDC4', linewidth=2)
            axes[0].plot(x_axis, results['validation_1'][loss_metric], 
                       label='Validation', color='#FF6B6B', linewidth=2)
            axes[0].set_xlabel('Boosting Rounds', fontsize=11)
            axes[0].set_ylabel(loss_metric.replace('_', ' ').title(), fontsize=11)
            axes[0].set_title(f'Training vs Validation {loss_metric.replace("_", " ").title()}', 
                            fontsize=12, fontweight='bold')
            axes[0].legend(loc='upper right')
            axes[0].grid(True, alpha=0.3)
        else:
            axes[0].text(0.5, 0.5, 'No loss metric available', 
                       ha='center', va='center', fontsize=12)
            axes[0].set_title('Loss Plot')
        
        # Plot 2: AUC-PR if available
        if 'aucpr' in available_metrics:
            epochs = len(results['validation_0']['aucpr'])
            x_axis = range(0, epochs)
            
            axes[1].plot(x_axis, results['validation_0']['aucpr'], 
                       label='Training', color='#4ECDC4', linewidth=2)
            axes[1].plot(x_axis, results['validation_1']['aucpr'], 
                       label='Validation', color='#FF6B6B', linewidth=2)
            axes[1].set_xlabel('Boosting Rounds', fontsize=11)
            axes[1].set_ylabel('PR-AUC Score', fontsize=11)
            axes[1].set_title('Training vs Validation PR-AUC', 
                            fontsize=12, fontweight='bold')
            axes[1].legend(loc='lower right')
            axes[1].grid(True, alpha=0.3)
        elif 'auc' in available_metrics:
            epochs = len(results['validation_0']['auc'])
            x_axis = range(0, epochs)
            
            axes[1].plot(x_axis, results['validation_0']['auc'], 
                       label='Training', color='#4ECDC4', linewidth=2)
            axes[1].plot(x_axis, results['validation_1']['auc'], 
                       label='Validation', color='#FF6B6B', linewidth=2)
            axes[1].set_xlabel('Boosting Rounds', fontsize=11)
            axes[1].set_ylabel('AUC Score', fontsize=11)
            axes[1].set_title('Training vs Validation AUC', 
                            fontsize=12, fontweight='bold')
            axes[1].legend(loc='lower right')
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'No AUC metric available', 
                       ha='center', va='center', fontsize=12)
            axes[1].set_title('AUC Plot')
        
        plt.tight_layout()
        os.makedirs('../models', exist_ok=True)
        plt.savefig('../models/training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print("✅ Training history plots saved to ../models/training_history.png")
    
    def save_model(self, path='../models/xgboost_fraud_model.pkl'):
        """Save trained model with metadata."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'best_params': self.best_params,
                'optimal_threshold': self.optimal_threshold
            }, f)
        print(f"\n✅ Model saved to {path}")
        print(f"   Optimal threshold: {self.optimal_threshold:.4f}")
    
    def load_model(self, path='../models/xgboost_fraud_model.pkl'):
        """Load trained model with metadata."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.best_params = data['best_params']
        self.optimal_threshold = data.get('optimal_threshold', 0.5)
        print(f"✅ Model loaded from {path}")
        print(f"   Optimal threshold: {self.optimal_threshold:.4f}")
        return self


# Test the model
if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from data_preprocessing import FraudDataPreprocessor
    
    # Generate and preprocess data
    print("\n" + "="*60)
    print("BANK FRAUD DETECTION - XGBOOST PIPELINE")
    print("="*60)
    
    print("\n🔹 Step 1: Generating synthetic transaction data...")
    preprocessor = FraudDataPreprocessor(n_samples=50000, fraud_ratio=0.02)
    df = preprocessor.generate_synthetic_data()
    X_train, X_test, y_train, y_test, num_cols, cat_cols = preprocessor.preprocess(df)
    
    # Train XGBoost with cost-sensitive learning
    print("\n🔹 Step 2: Training XGBoost with Cost-Sensitive Learning...")
    xgb_model = FraudDetectionXGBoost()
    xgb_model.train_with_cost_sensitive(X_train, y_train, X_test, y_test, 
                                         feature_names=X_train.columns.tolist(),
                                         cost_ratio=10)
    
    # Evaluate with optimal threshold
    print("\n🔹 Step 3: Evaluating model performance...")
    results = xgb_model.evaluate(X_test, y_test)
    
    # Feature importance analysis
    print("\n🔹 Step 4: Analyzing feature importance...")
    importance_df = xgb_model.plot_feature_importance(top_n=15)
    
    # Plot training history
    print("\n🔹 Step 5: Plotting training history...")
    xgb_model.plot_training_history()
    
    # Save model
    print("\n🔹 Step 6: Saving trained model...")
    xgb_model.save_model()
    
    print("\n" + "="*60)
    print("✅ XGBOOST FRAUD DETECTION PIPELINE COMPLETE!")
    print("="*60)
    print(f"\n📁 Files created:")
    print(f"   - models/xgboost_fraud_model.pkl")
    print(f"   - models/feature_importance_xgboost.png")
    print(f"   - models/training_history.png")
    print(f"   - data/raw/raw_transactions.csv")