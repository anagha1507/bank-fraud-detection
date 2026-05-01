import numpy as np
import pandas as pd
import pickle
import os
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SHAPFraudExplainer:
    """
    SHAP-based explainability for fraud detection model.
    Explains individual predictions and global feature importance.
    """
    
    def __init__(self, model, feature_names, X_train_sample=None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained XGBoost model
            feature_names: List of feature names
            X_train_sample: Sample of training data for background distribution
        """
        self.model = model
        self.feature_names = feature_names
        self.X_train_sample = X_train_sample
        self.explainer = None
        self.shap_values = None
        
    def create_explainer(self, X_background, method='tree'):
        """
        Create SHAP explainer.
        
        Args:
            X_background: Background dataset for SHAP (usually training sample)
            method: 'tree' for tree models, 'kernel' for others
        """
        print("\n" + "="*60)
        print("🔮 CREATING SHAP EXPLAINER")
        print("="*60)
        
        print(f"Using {len(X_background)} samples as background distribution...")
        
        if method == 'tree':
            # TreeExplainer is optimized for XGBoost
            self.explainer = shap.TreeExplainer(
                self.model,
                data=X_background,
                feature_names=self.feature_names
            )
            print("✅ TreeExplainer created (optimized for XGBoost)")
        else:
            # KernelExplainer for other models
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                X_background,
                feature_names=self.feature_names
            )
            print("✅ KernelExplainer created")
        
        return self.explainer
    
    def compute_shap_values(self, X_sample, max_display=1000):
        """
        Compute SHAP values for a sample of transactions.
        
        Args:
            X_sample: Data to compute SHAP values for
            max_display: Maximum samples to compute (SHAP can be slow)
        """
        print("\n" + "="*60)
        print("📊 COMPUTING SHAP VALUES")
        print("="*60)
        
        # Limit sample size for faster computation
        if len(X_sample) > max_display:
            print(f"Sampling {max_display} transactions for SHAP analysis...")
            X_shap = X_sample.sample(n=max_display, random_state=42)
        else:
            X_shap = X_sample.copy()
            print(f"Computing SHAP for all {len(X_sample)} transactions...")
        
        # Compute SHAP values
        print("Computing SHAP values (this may take a minute)...")
        self.shap_values = self.explainer.shap_values(X_shap)
        
        print(f"✅ SHAP values computed!")
        print(f"   Shape: {self.shap_values.shape}")
        
        return self.shap_values, X_shap
    
    def plot_global_feature_importance(self, X_sample, save_path='../models/shap_summary.png'):
        """
        Generate SHAP summary plot (global feature importance).
        Shows which features drive predictions and in which direction.
        """
        print("\n" + "="*60)
        print("🌍 GLOBAL FEATURE IMPORTANCE (SHAP)")
        print("="*60)
        
        if self.shap_values is None:
            _, X_sample = self.compute_shap_values(X_sample)
        
        # Create figure
        plt.figure(figsize=(14, 10))
        
        # SHAP Summary Plot - Beeswarm
        shap.summary_plot(
            self.shap_values,
            X_sample,
            feature_names=self.feature_names,
            max_display=20,
            show=False,
            plot_size=(14, 10)
        )
        
        plt.title('SHAP Feature Importance - Global Impact on Fraud Detection', 
                 fontsize=16, fontweight='bold', pad=20)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Global SHAP summary saved to {save_path}")
        
        # Also create bar plot version
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            self.shap_values,
            X_sample,
            feature_names=self.feature_names,
            plot_type='bar',
            max_display=20,
            show=False
        )
        plt.title('SHAP Feature Importance (Mean Impact)', 
                 fontsize=14, fontweight='bold')
        plt.tight_layout()
        bar_path = save_path.replace('.png', '_bar.png')
        plt.savefig(bar_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ SHAP bar plot saved to {bar_path}")
    
    def explain_single_prediction(self, transaction, transaction_index=0, 
                                   save_path='../models/shap_single_explanation.png'):
        """
        Explain a single transaction prediction in detail.
        
        Args:
            transaction: Single transaction dataframe row
            transaction_index: Index for display
            save_path: Path to save the explanation plot
        """
        print("\n" + "="*60)
        print(f"🔍 EXPLAINING SINGLE TRANSACTION #{transaction_index}")
        print("="*60)
        
        # Get prediction
        if hasattr(transaction, 'values'):
            X_single = transaction.values.reshape(1, -1)
        else:
            X_single = np.array(transaction).reshape(1, -1)
        
        # Predict
        fraud_probability = self.model.predict_proba(X_single)[0, 1]
        prediction = "🚨 FRAUD" if fraud_probability >= 0.5 else "✅ NORMAL"
        
        print(f"\n📊 Prediction: {prediction}")
        print(f"   Fraud Probability: {fraud_probability:.4f} ({fraud_probability*100:.2f}%)")
        
        # Compute SHAP values for this prediction
        shap_values_single = self.explainer.shap_values(X_single)
        
        # Create explanation
        plt.figure(figsize=(14, 8))
        
        # Waterfall plot
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_values_single[0],
                base_values=self.explainer.expected_value,
                data=X_single[0],
                feature_names=self.feature_names
            ),
            max_display=15,
            show=False
        )
        
        plt.title(f'Transaction #{transaction_index} - Fraud Probability: {fraud_probability:.2%}',
                 fontsize=14, fontweight='bold', pad=20)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        # Print top contributing features
        print(f"\n📈 Top Features Driving This Prediction:")
        print("-" * 50)
        
        # Create dataframe of feature contributions
        contributions = pd.DataFrame({
            'feature': self.feature_names,
            'value': X_single[0],
            'shap_value': shap_values_single[0]
        })
        contributions['abs_shap'] = np.abs(contributions['shap_value'])
        contributions = contributions.sort_values('abs_shap', ascending=False)
        
        for i, (_, row) in enumerate(contributions.head(10).iterrows()):
            direction = "🔺 Increases fraud risk" if row['shap_value'] > 0 else "🔻 Decreases fraud risk"
            print(f"  {i+1}. {row['feature']:<25} | SHAP: {row['shap_value']:+.4f} | {direction}")
        
        print(f"\n✅ Single prediction explanation saved to {save_path}")
        
        return {
            'prediction': fraud_probability,
            'is_fraud': fraud_probability >= 0.5,
            'top_features': contributions.head(10)
        }
    
    def analyze_false_negatives(self, X_test, y_test, n_examples=3):
        """
        Analyze false negative cases (fraud predicted as normal).
        This helps understand what the model is missing.
        """
        print("\n" + "="*60)
        print("🚨 ANALYZING FALSE NEGATIVES (MISSED FRAUD)")
        print("="*60)
        
        # Get predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Find false negatives
        fn_mask = (y_test == 1) & (y_pred == 0)
        fn_indices = np.where(fn_mask)[0]
        
        if len(fn_indices) == 0:
            print("✅ No false negatives found! Model caught all fraud.")
            return []
        
        print(f"Found {len(fn_indices)} missed fraud cases")
        print(f"Analyzing top {min(n_examples, len(fn_indices))} cases...\n")
        
        fn_analyses = []
        for i, idx in enumerate(fn_indices[:n_examples]):
            print(f"{'='*50}")
            print(f"Missed Fraud Case #{i+1} (Index: {idx})")
            print(f"{'='*50}")
            
            transaction = X_test.iloc[idx:idx+1]
            fraud_prob = y_pred_proba[idx]
            
            print(f"Fraud Probability: {fraud_prob:.4f} (below threshold)")
            print(f"Actual: FRAUD | Predicted: NORMAL")
            
            # Get SHAP explanation
            shap_vals = self.explainer.shap_values(transaction)
            
            contributions = pd.DataFrame({
                'feature': self.feature_names,
                'value': transaction.values[0],
                'shap_value': shap_vals[0]
            })
            contributions['abs_shap'] = np.abs(contributions['shap_value'])
            contributions = contributions.sort_values('shap_value', ascending=True)
            
            print("\nTop features that decreased fraud score (led to missing this fraud):")
            for j, (_, row) in enumerate(contributions.head(5).iterrows()):
                print(f"  {row['feature']:<25} | SHAP: {row['shap_value']:+.4f} | Value: {row['value']:.4f}")
            
            fn_analyses.append({
                'index': idx,
                'fraud_probability': fraud_prob,
                'contributions': contributions
            })
            
            # Waterfall plot
            plt.figure(figsize=(14, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals[0],
                    base_values=self.explainer.expected_value,
                    data=transaction.values[0],
                    feature_names=self.feature_names
                ),
                max_display=10,
                show=False
            )
            plt.title(f'False Negative Analysis - Case #{i+1}\nFraud Probability: {fraud_prob:.2%}',
                     fontsize=12, fontweight='bold')
            
            fn_path = f'../models/false_negative_case_{i+1}.png'
            os.makedirs(os.path.dirname(fn_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(fn_path, dpi=150, bbox_inches='tight')
            plt.show()
        
        return fn_analyses
    
    def analyze_false_positives(self, X_test, y_test, n_examples=3):
        """
        Analyze false positive cases (normal predicted as fraud).
        This helps reduce false alarms.
        """
        print("\n" + "="*60)
        print("⚠️  ANALYZING FALSE POSITIVES (FALSE ALARMS)")
        print("="*60)
        
        # Get predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        # Find false positives
        fp_mask = (y_test == 0) & (y_pred == 1)
        fp_indices = np.where(fp_mask)[0]
        
        if len(fp_indices) == 0:
            print("✅ No false positives found! Perfect precision.")
            return []
        
        print(f"Found {len(fp_indices)} false alarms")
        print(f"Analyzing top {min(n_examples, len(fp_indices))} cases...\n")
        
        fp_analyses = []
        for i, idx in enumerate(fp_indices[:n_examples]):
            print(f"{'='*50}")
            print(f"False Alarm Case #{i+1} (Index: {idx})")
            print(f"{'='*50}")
            
            transaction = X_test.iloc[idx:idx+1]
            fraud_prob = y_pred_proba[idx]
            
            print(f"Fraud Probability: {fraud_prob:.4f} (above threshold)")
            print(f"Actual: NORMAL | Predicted: FRAUD")
            
            # Get SHAP explanation
            shap_vals = self.explainer.shap_values(transaction)
            
            contributions = pd.DataFrame({
                'feature': self.feature_names,
                'value': transaction.values[0],
                'shap_value': shap_vals[0]
            })
            contributions['abs_shap'] = np.abs(contributions['shap_value'])
            contributions = contributions.sort_values('shap_value', ascending=False)
            
            print("\nTop features that increased fraud score (caused false alarm):")
            for j, (_, row) in enumerate(contributions.head(5).iterrows()):
                print(f"  {row['feature']:<25} | SHAP: {row['shap_value']:+.4f} | Value: {row['value']:.4f}")
            
            fp_analyses.append({
                'index': idx,
                'fraud_probability': fraud_prob,
                'contributions': contributions
            })
            
            # Waterfall plot
            plt.figure(figsize=(14, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_vals[0],
                    base_values=self.explainer.expected_value,
                    data=transaction.values[0],
                    feature_names=self.feature_names
                ),
                max_display=10,
                show=False
            )
            plt.title(f'False Positive Analysis - Case #{i+1}\nFraud Probability: {fraud_prob:.2%}',
                     fontsize=12, fontweight='bold')
            
            fp_path = f'../models/false_positive_case_{i+1}.png'
            os.makedirs(os.path.dirname(fp_path), exist_ok=True)
            plt.tight_layout()
            plt.savefig(fp_path, dpi=150, bbox_inches='tight')
            plt.show()
        
        return fp_analyses
    
    def generate_dependence_plots(self, X_sample, top_features=None, n_features=5):
        """
        Generate SHAP dependence plots for top features.
        Shows how feature values affect SHAP values.
        """
        print("\n" + "="*60)
        print("📈 SHAP DEPENDENCE PLOTS")
        print("="*60)
        
        if self.shap_values is None:
            self.shap_values, X_sample = self.compute_shap_values(X_sample)
        
        # Get top features by mean absolute SHAP value
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        if top_features is None:
            top_features = feature_importance['feature'].head(n_features).tolist()
        
        # Create dependence plots
        n_cols = min(3, len(top_features))
        n_rows = (len(top_features) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        axes = axes.flatten() if hasattr(axes, 'flatten') else np.array([axes])
        
        for i, feature in enumerate(top_features):
            if i < len(axes):
                feature_idx = list(self.feature_names).index(feature)
                
                shap.dependence_plot(
                    feature_idx,
                    self.shap_values,
                    X_sample.values,
                    feature_names=self.feature_names,
                    ax=axes[i],
                    show=False
                )
                axes[i].set_title(f'SHAP Dependence: {feature}', fontsize=11, fontweight='bold')
        
        # Hide empty subplots
        for j in range(i+1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle('SHAP Dependence Plots - How Features Affect Predictions',
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        dep_path = '../models/shap_dependence_plots.png'
        os.makedirs(os.path.dirname(dep_path), exist_ok=True)
        plt.savefig(dep_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Dependence plots saved to {dep_path}")
        
        return feature_importance
    
    def generate_report(self, X_test, y_test, X_train_sample):
        """
        Generate comprehensive SHAP analysis report.
        """
        print("\n" + "="*60)
        print("📋 GENERATING COMPREHENSIVE SHAP REPORT")
        print("="*60)
        
        report = []
        report.append("="*80)
        report.append("BANK FRAUD DETECTION - SHAP EXPLAINABILITY REPORT")
        report.append("="*80)
        report.append("")
        
        # 1. Model Overview
        report.append("1. MODEL OVERVIEW")
        report.append("-" * 40)
        report.append(f"Model Type: XGBoost Classifier")
        report.append(f"Number of Features: {len(self.feature_names)}")
        report.append(f"Training Samples: {len(X_train_sample):,}")
        report.append(f"Test Samples: {len(X_test):,}")
        report.append("")
        
        # 2. Top Features
        report.append("2. TOP FEATURES BY SHAP IMPORTANCE")
        report.append("-" * 40)
        
        mean_shap = np.abs(self.shap_values).mean(axis=0)
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'mean_abs_shap': mean_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
            report.append(f"  {i+1:2d}. {row['feature']:<25} | SHAP Importance: {row['mean_abs_shap']:.4f}")
        
        report.append("")
        
        # 3. Prediction Distribution
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        report.append("3. PREDICTION DISTRIBUTION")
        report.append("-" * 40)
        report.append(f"Mean Fraud Probability: {y_pred_proba.mean():.4f}")
        report.append(f"Median Fraud Probability: {np.median(y_pred_proba):.4f}")
        report.append(f"Std Fraud Probability: {y_pred_proba.std():.4f}")
        report.append(f"Min Fraud Probability: {y_pred_proba.min():.4f}")
        report.append(f"Max Fraud Probability: {y_pred_proba.max():.4f}")
        report.append("")
        
        # 4. False Negative Analysis
        y_pred = (y_pred_proba >= 0.5).astype(int)
        fn_count = ((y_test == 1) & (y_pred == 0)).sum()
        fp_count = ((y_test == 0) & (y_pred == 1)).sum()
        
        report.append("4. ERROR ANALYSIS")
        report.append("-" * 40)
        report.append(f"False Negatives (Missed Fraud): {fn_count}")
        report.append(f"False Positives (False Alarms): {fp_count}")
        report.append(f"Total Errors: {fn_count + fp_count}")
        report.append("")
        
        # 5. Recommendations
        report.append("5. RECOMMENDATIONS")
        report.append("-" * 40)
        report.append("• Review top SHAP features for feature engineering improvements")
        report.append("• Investigate false negatives for patterns the model misses")
        report.append("• Use SHAP explanations in fraud analyst workflow")
        report.append("• Monitor feature distributions for drift over time")
        report.append("")
        
        report.append("="*80)
        report.append("END OF REPORT")
        report.append("="*80)
        
        # Save report
        report_text = "\n".join(report)
        report_path = '../models/shap_analysis_report.txt'
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n✅ Report saved to {report_path}")
        
        return report_text


# Main execution
if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from data_preprocessing import FraudDataPreprocessor
    from xgboost_model import FraudDetectionXGBoost
    
    print("\n" + "="*60)
    print("🔮 SHAP EXPLAINABILITY PIPELINE")
    print("="*60)
    
    # Step 1: Load or create data
    print("\n🔹 Step 1: Preparing data...")
    preprocessor = FraudDataPreprocessor(n_samples=50000, fraud_ratio=0.02)
    df = preprocessor.generate_synthetic_data()
    X_train, X_test, y_train, y_test, num_cols, cat_cols = preprocessor.preprocess(df)
    
    # Step 2: Train model (or load pre-trained)
    print("\n🔹 Step 2: Training model...")
    xgb_detector = FraudDetectionXGBoost()
    
    # Check if model already exists
    model_path = '../models/xgboost_fraud_model.pkl'
    if os.path.exists(model_path):
        print("Loading pre-trained model...")
        xgb_detector.load_model(model_path)
    else:
        xgb_detector.train_with_cost_sensitive(
            X_train, y_train, X_test, y_test,
            feature_names=X_train.columns.tolist(),
            cost_ratio=10
        )
        xgb_detector.save_model(model_path)
    
    # Step 3: Create SHAP explainer
    print("\n🔹 Step 3: Creating SHAP explainer...")
    shap_explainer = SHAPFraudExplainer(
        model=xgb_detector.model,
        feature_names=X_train.columns.tolist(),
        X_train_sample=X_train.sample(n=500, random_state=42)
    )
    
    # Use a sample of training data as background
    X_background = X_train.sample(n=200, random_state=42)
    shap_explainer.create_explainer(X_background, method='tree')
    
    # Step 4: Compute SHAP values for test set
    print("\n🔹 Step 4: Computing SHAP values...")
    X_test_sample = X_test.sample(n=500, random_state=42)
    shap_values, X_shap = shap_explainer.compute_shap_values(X_test_sample, max_display=500)
    
    # Step 5: Global feature importance
    print("\n🔹 Step 5: Generating global feature importance...")
    shap_explainer.plot_global_feature_importance(X_shap)
    
    # Step 6: Explain individual predictions
    print("\n🔹 Step 6: Explaining individual predictions...")
    
    # Explain a fraud case
    fraud_indices = y_test[y_test == 1].index[:3]
    for i, idx in enumerate(fraud_indices):
        transaction = X_test.loc[idx:idx]
        shap_explainer.explain_single_prediction(
            transaction,
            transaction_index=idx,
            save_path=f'../models/shap_fraud_explanation_{i+1}.png'
        )
    
    # Explain a normal case
    normal_indices = y_test[y_test == 0].index[:2]
    for i, idx in enumerate(normal_indices):
        transaction = X_test.loc[idx:idx]
        shap_explainer.explain_single_prediction(
            transaction,
            transaction_index=idx,
            save_path=f'../models/shap_normal_explanation_{i+1}.png'
        )
    
    # Step 7: Analyze errors
    print("\n🔹 Step 7: Analyzing model errors...")
    shap_explainer.analyze_false_negatives(X_test, y_test, n_examples=2)
    shap_explainer.analyze_false_positives(X_test, y_test, n_examples=2)
    
    # Step 8: Dependence plots
    print("\n🔹 Step 8: Generating dependence plots...")
    shap_explainer.generate_dependence_plots(X_shap, n_features=6)
    
    # Step 9: Generate report
    print("\n🔹 Step 9: Generating comprehensive report...")
    shap_explainer.generate_report(X_test, y_test, X_train)
    
    print("\n" + "="*60)
    print("✅ SHAP EXPLAINABILITY PIPELINE COMPLETE!")
    print("="*60)
    print("\n📁 Generated Files:")
    print("   - models/shap_summary.png (Global feature importance)")
    print("   - models/shap_summary_bar.png (Bar chart version)")
    print("   - models/shap_fraud_explanation_*.png (Fraud case explanations)")
    print("   - models/shap_normal_explanation_*.png (Normal case explanations)")
    print("   - models/false_negative_case_*.png (Missed fraud analysis)")
    print("   - models/false_positive_case_*.png (False alarm analysis)")
    print("   - models/shap_dependence_plots.png (Feature dependence)")
    print("   - models/shap_analysis_report.txt (Text report)")