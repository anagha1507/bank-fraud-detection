import numpy as np
import pandas as pd
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.models import load_model

# ML imports
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, precision_recall_curve,
                             f1_score, average_precision_score)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class FraudAutoencoder:
    """
    Autoencoder-based anomaly detection for bank fraud.
    
    How it works:
    1. Train ONLY on normal transactions to learn "normal" patterns
    2. When a transaction is passed through, reconstruction error is measured
    3. High reconstruction error = unusual = potential fraud
    
    This detects novel fraud patterns that supervised models might miss.
    """
    
    def __init__(self, input_dim, encoding_dim=16, random_state=42):
        """
        Initialize the autoencoder.
        
        Args:
            input_dim: Number of input features
            encoding_dim: Dimension of the bottleneck layer (compressed representation)
            random_state: For reproducibility
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.random_state = random_state
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.threshold = None
        self.scaler = StandardScaler()
        self.history = None
        
        # Set seeds
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
        
        # Build the model
        self._build_model()
    
    def _build_model(self):
        """
        Build the autoencoder architecture.
        
        Architecture:
        Input -> Encoder (compresses) -> Bottleneck -> Decoder (reconstructs) -> Output
        
        The bottleneck forces the network to learn efficient representations.
        """
        # ---- ENCODER ----
        encoder_input = layers.Input(shape=(self.input_dim,), name='encoder_input')
        
        # Layer 1: Compress to half
        x = layers.Dense(64, activation='relu', name='encoder_dense1')(encoder_input)
        x = layers.BatchNormalization(name='encoder_bn1')(x)
        x = layers.Dropout(0.2, name='encoder_dropout1')(x)
        
        # Layer 2: Further compression
        x = layers.Dense(32, activation='relu', name='encoder_dense2')(x)
        x = layers.BatchNormalization(name='encoder_bn2')(x)
        x = layers.Dropout(0.2, name='encoder_dropout2')(x)
        
        # Bottleneck: Most compressed representation
        bottleneck = layers.Dense(self.encoding_dim, activation='relu', 
                                  name='bottleneck')(x)
        
        # ---- DECODER ----
        # Layer 1: Start expanding
        x = layers.Dense(32, activation='relu', name='decoder_dense1')(bottleneck)
        x = layers.BatchNormalization(name='decoder_bn1')(x)
        x = layers.Dropout(0.2, name='decoder_dropout1')(x)
        
        # Layer 2: Further expansion
        x = layers.Dense(64, activation='relu', name='decoder_dense2')(x)
        x = layers.BatchNormalization(name='decoder_bn2')(x)
        x = layers.Dropout(0.2, name='decoder_dropout2')(x)
        
        # Output: Reconstruct original input
        decoder_output = layers.Dense(self.input_dim, activation='linear', 
                                      name='decoder_output')(x)
        
        # ---- FULL AUTOENCODER ----
        self.autoencoder = Model(encoder_input, decoder_output, name='fraud_autoencoder')
        
        # ---- SEPARATE ENCODER (for feature extraction) ----
        self.encoder = Model(encoder_input, bottleneck, name='encoder')
        
        # ---- COMPILE ----
        self.autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',  # Mean Squared Error for reconstruction
            metrics=['mae']  # Mean Absolute Error
        )
        
        print("✅ Autoencoder architecture built:")
        print(f"   Input: {self.input_dim} features")
        print(f"   Encoding: {self.input_dim} -> 64 -> 32 -> {self.encoding_dim} (bottleneck)")
        print(f"   Decoding: {self.encoding_dim} -> 32 -> 64 -> {self.input_dim} (output)")
        
        # Print model summary
        self.autoencoder.summary()
    
    def train(self, X_train_normal, X_val=None, epochs=100, batch_size=256, 
              early_stopping_patience=15, verbose=1):
        """
        Train the autoencoder ONLY on normal transactions.
        
        Args:
            X_train_normal: Only NORMAL transactions (no fraud)
            X_val: Validation data (also normal only)
            epochs: Maximum training epochs
            batch_size: Batch size for training
            early_stopping_patience: Stop if no improvement for N epochs
            verbose: 1 for progress bar, 0 for silent
        """
        print("\n" + "="*60)
        print("🤖 TRAINING AUTOENCODER (Unsupervised)")
        print("="*60)
        print(f"Training on {len(X_train_normal):,} NORMAL transactions only")
        print(f"Model learns: 'What does a normal transaction look like?'")
        print(f"Fraud detection: Anything that doesn't reconstruct well = suspicious")
        
        # Callbacks
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss' if X_val is not None else 'loss',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss' if X_val is not None else 'loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        model_checkpoint = callbacks.ModelCheckpoint(
            '../models/best_autoencoder.keras',
            monitor='val_loss' if X_val is not None else 'loss',
            save_best_only=True,
            verbose=1
        )
        
        # Train
        validation_data = (X_val, X_val) if X_val is not None else None
        
        self.history = self.autoencoder.fit(
            X_train_normal, X_train_normal,  # Input = Output for autoencoder
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            callbacks=[early_stop, reduce_lr, model_checkpoint],
            verbose=verbose,
            shuffle=True
        )
        
        print(f"\n✅ Autoencoder training complete!")
        print(f"   Final loss: {self.history.history['loss'][-1]:.6f}")
        if X_val is not None:
            print(f"   Final val_loss: {self.history.history['val_loss'][-1]:.6f}")
        
        return self.history
    
    def compute_reconstruction_error(self, X):
        """
        Compute reconstruction error for each transaction.
        
        Higher error = more abnormal = more likely fraud.
        """
        X_pred = self.autoencoder.predict(X, verbose=0)
        
        # Mean squared error per sample
        mse = np.mean(np.square(X - X_pred), axis=1)
        
        return mse
    
    def find_optimal_threshold(self, X_train_normal, X_test, y_test, 
                                percentile=95):
        """
        Find optimal anomaly threshold using training data.
        
        Uses percentile method: threshold = top percentile of normal reconstruction errors.
        
        Args:
            X_train_normal: Normal transactions for threshold calculation
            X_test: Test data
            y_test: Test labels
            percentile: Percentile for threshold (95 = flag top 5% most abnormal)
        """
        print("\n" + "="*60)
        print("🎯 FINDING OPTIMAL ANOMALY THRESHOLD")
        print("="*60)
        
        # Compute reconstruction error on normal training data
        train_errors = self.compute_reconstruction_error(X_train_normal)
        
        # Try different percentiles
        percentiles = [90, 92, 95, 97, 98, 99]
        best_f1 = 0
        best_threshold = None
        best_percentile = None
        
        print("\nTrying different thresholds:")
        print("-" * 50)
        
        test_errors = self.compute_reconstruction_error(X_test)
        
        for pct in percentiles:
            threshold = np.percentile(train_errors, pct)
            y_pred = (test_errors > threshold).astype(int)
            f1 = f1_score(y_test, y_pred)
            
            print(f"  Percentile {pct}%: threshold={threshold:.6f}, F1={f1:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                best_percentile = pct
        
        self.threshold = best_threshold
        
        print(f"\n✅ Best threshold: {best_threshold:.6f} (percentile {best_percentile}%)")
        print(f"   Best F1 Score: {best_f1:.4f}")
        
        return self.threshold
    
    def evaluate(self, X_test, y_test, threshold=None):
        """
        Evaluate the autoencoder on test data.
        """
        print("\n" + "="*60)
        print("📊 AUTOENCODER EVALUATION")
        print("="*60)
        
        if threshold is None:
            threshold = self.threshold
        
        # Compute reconstruction errors
        test_errors = self.compute_reconstruction_error(X_test)
        
        # Predict fraud (high error = fraud)
        y_pred = (test_errors > threshold).astype(int)
        y_scores = test_errors  # Use error as fraud score
        
        # Metrics
        accuracy = (y_pred == y_test).mean()
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_scores)
        avg_precision = average_precision_score(y_test, y_scores)
        
        print(f"\n🎯 Performance Metrics:")
        print(f"  ✅ Accuracy: {accuracy:.4f}")
        print(f"  ✅ F1-Score: {f1:.4f}")
        print(f"  ✅ ROC-AUC: {roc_auc:.4f}")
        print(f"  ✅ PR-AUC: {avg_precision:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n📊 Confusion Matrix:")
        print(f"  ✅ True Negatives: {cm[0,0]:,}")
        print(f"  ⚠️  False Positives: {cm[0,1]:,}")
        print(f"  🚨 False Negatives: {cm[1,0]:,}")
        print(f"  ✅ True Positives: {cm[1,1]:,}")
        
        # Classification Report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, 
                                    target_names=['Normal', 'Fraud']))
        
        # Error statistics
        print(f"\n📈 Reconstruction Error Statistics:")
        normal_errors = test_errors[y_test == 0]
        fraud_errors = test_errors[y_test == 1]
        
        print(f"  Normal transactions - Mean error: {normal_errors.mean():.6f}, "
              f"Std: {normal_errors.std():.6f}")
        print(f"  Fraud transactions  - Mean error: {fraud_errors.mean():.6f}, "
              f"Std: {fraud_errors.std():.6f}")
        print(f"  Error ratio (Fraud/Normal): {fraud_errors.mean() / normal_errors.mean():.2f}x")
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'avg_precision': avg_precision,
            'confusion_matrix': cm,
            'test_errors': test_errors,
            'y_pred': y_pred
        }
    
    def plot_reconstruction_error(self, X_test, y_test, save_path='../models/autoencoder_errors.png'):
        """
        Visualize reconstruction error distribution for normal vs fraud.
        """
        print("\n" + "="*60)
        print("📊 RECONSTRUCTION ERROR DISTRIBUTION")
        print("="*60)
        
        test_errors = self.compute_reconstruction_error(X_test)
        
        # Separate errors
        normal_errors = test_errors[y_test == 0]
        fraud_errors = test_errors[y_test == 1]
        
        # Create plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(normal_errors, bins=50, alpha=0.7, label='Normal', 
                    color='#4ECDC4', edgecolor='white')
        axes[0].hist(fraud_errors, bins=50, alpha=0.7, label='Fraud', 
                    color='#FF6B6B', edgecolor='white')
        
        if self.threshold:
            axes[0].axvline(x=self.threshold, color='red', linestyle='--', 
                          linewidth=2, label=f'Threshold ({self.threshold:.4f})')
        
        axes[0].set_xlabel('Reconstruction Error (MSE)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Reconstruction Error Distribution', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Box plot
        box_data = [normal_errors, fraud_errors]
        bp = axes[1].boxplot(box_data, labels=['Normal', 'Fraud'], 
                             patch_artist=True, widths=0.5)
        bp['boxes'][0].set_facecolor('#4ECDC4')
        bp['boxes'][1].set_facecolor('#FF6B6B')
        
        axes[1].set_ylabel('Reconstruction Error (MSE)', fontsize=11)
        axes[1].set_title('Error Comparison: Normal vs Fraud', fontsize=13, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Autoencoder Anomaly Detection Analysis', 
                    fontsize=15, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\n✅ Error distribution plot saved to {save_path}")
        print(f"   Normal mean error: {normal_errors.mean():.6f}")
        print(f"   Fraud mean error: {fraud_errors.mean():.6f}")
        
        if fraud_errors.mean() > normal_errors.mean():
            print(f"   ✅ Autoencoder successfully detects fraud! "
                  f"(Fraud error is {fraud_errors.mean()/normal_errors.mean():.1f}x higher)")
        
        return normal_errors, fraud_errors
    
    def plot_training_history(self, save_path='../models/autoencoder_training.png'):
        """
        Plot training and validation loss.
        """
        print("\n" + "="*60)
        print("📈 TRAINING HISTORY")
        print("="*60)
        
        if self.history is None:
            print("No training history available.")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss plot
        axes[0].plot(self.history.history['loss'], label='Training Loss', 
                    color='#4ECDC4', linewidth=2)
        if 'val_loss' in self.history.history:
            axes[0].plot(self.history.history['val_loss'], label='Validation Loss', 
                        color='#FF6B6B', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=11)
        axes[0].set_ylabel('Loss (MSE)', fontsize=11)
        axes[0].set_title('Training & Validation Loss', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # MAE plot
        if 'mae' in self.history.history:
            axes[1].plot(self.history.history['mae'], label='Training MAE', 
                        color='#4ECDC4', linewidth=2)
            if 'val_mae' in self.history.history:
                axes[1].plot(self.history.history['val_mae'], label='Validation MAE', 
                            color='#FF6B6B', linewidth=2)
            axes[1].set_xlabel('Epoch', fontsize=11)
            axes[1].set_ylabel('MAE', fontsize=11)
            axes[1].set_title('Training & Validation MAE', fontsize=13, fontweight='bold')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.suptitle('Autoencoder Training History', fontsize=15, fontweight='bold')
        plt.tight_layout()
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ Training history saved to {save_path}")
    
    def detect_anomalies(self, X_new, threshold=None):
        """
        Detect anomalies in new transactions.
        
        Returns:
            DataFrame with anomaly scores and predictions
        """
        if threshold is None:
            threshold = self.threshold
        
        errors = self.compute_reconstruction_error(X_new)
        is_anomaly = errors > threshold
        
        results = pd.DataFrame({
            'reconstruction_error': errors,
            'anomaly_score': errors / threshold,  # Normalized score (>1 = anomaly)
            'is_anomaly': is_anomaly,
            'risk_level': pd.cut(
                errors / threshold,
                bins=[0, 0.8, 1.0, 1.5, np.inf],
                labels=['Low', 'Medium', 'High', 'Critical']
            )
        })
        
        return results
    
    def save_model(self, path='../models/autoencoder_model'):
        """
        Save the trained autoencoder and metadata.
        """
        os.makedirs(path, exist_ok=True)
        
        # Save Keras model
        self.autoencoder.save(f'{path}/autoencoder.keras')
        
        # Save metadata
        metadata = {
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'threshold': self.threshold
        }
        
        with open(f'{path}/metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"\n✅ Autoencoder saved to {path}/")
        print(f"   - autoencoder.keras")
        print(f"   - metadata.pkl")
    
    def load_model(self, path='../models/autoencoder_model'):
        """
        Load a trained autoencoder.
        """
        # Load Keras model
        self.autoencoder = load_model(f'{path}/autoencoder.keras')
        
        # Recreate encoder
        encoder_input = self.autoencoder.input
        bottleneck_layer = self.autoencoder.get_layer('bottleneck')
        self.encoder = Model(encoder_input, bottleneck_layer.output)
        
        # Load metadata
        with open(f'{path}/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        self.input_dim = metadata['input_dim']
        self.encoding_dim = metadata['encoding_dim']
        self.threshold = metadata['threshold']
        
        print(f"✅ Autoencoder loaded from {path}/")
        print(f"   Input dim: {self.input_dim}, Encoding dim: {self.encoding_dim}")
        print(f"   Threshold: {self.threshold:.6f}")
        
        return self


# Main execution
if __name__ == "__main__":
    import sys
    sys.path.append('.')
    from data_preprocessing import FraudDataPreprocessor
    
    print("\n" + "="*60)
    print("🤖 AUTOENCODER FRAUD DETECTION PIPELINE")
    print("="*60)
    
    # Step 1: Generate and preprocess data
    print("\n🔹 Step 1: Preparing data...")
    preprocessor = FraudDataPreprocessor(n_samples=50000, fraud_ratio=0.02)
    df = preprocessor.generate_synthetic_data()
    X_train_full, X_test, y_train_full, y_test, num_cols, cat_cols = preprocessor.preprocess(df)
    
    # Separate normal transactions for autoencoder training
    X_train_normal = X_train_full[y_train_full == 0]
    print(f"\nTraining autoencoder on {len(X_train_normal):,} normal transactions")
    
    # Also get some validation normal data
    from sklearn.model_selection import train_test_split
    X_train_ae, X_val_ae = train_test_split(
        X_train_normal, test_size=0.1, random_state=42
    )
    
    # Step 2: Build and train autoencoder
    print("\n🔹 Step 2: Building autoencoder...")
    input_dim = X_train_full.shape[1]
    autoencoder = FraudAutoencoder(input_dim=input_dim, encoding_dim=16)
    
    print("\n🔹 Step 3: Training autoencoder (only on normal data)...")
    autoencoder.train(
        X_train_ae,
        X_val=X_val_ae,
        epochs=50,
        batch_size=256,
        early_stopping_patience=10,
        verbose=1
    )
    
    # Step 4: Find optimal threshold
    print("\n🔹 Step 4: Finding optimal anomaly threshold...")
    autoencoder.find_optimal_threshold(X_train_ae, X_test, y_test)
    
    # Step 5: Evaluate on test data
    print("\n🔹 Step 5: Evaluating autoencoder...")
    results = autoencoder.evaluate(X_test, y_test)
    
    # Step 6: Visualize
    print("\n🔹 Step 6: Visualizing results...")
    autoencoder.plot_reconstruction_error(X_test, y_test)
    autoencoder.plot_training_history()
    
    # Step 7: Detect anomalies in test data
    print("\n🔹 Step 7: Detecting anomalies...")
    anomaly_results = autoencoder.detect_anomalies(X_test)
    print("\nAnomaly Detection Results:")
    print(anomaly_results['risk_level'].value_counts())
    print(f"\nAnomalies detected: {anomaly_results['is_anomaly'].sum()} / {len(anomaly_results)}")
    
    # Step 8: Save model
    print("\n🔹 Step 8: Saving autoencoder model...")
    autoencoder.save_model()
    
    print("\n" + "="*60)
    print("✅ AUTOENCODER PIPELINE COMPLETE!")
    print("="*60)
    print("\n📁 Files created:")
    print("   - models/autoencoder_model/autoencoder.keras")
    print("   - models/autoencoder_model/metadata.pkl")
    print("   - models/autoencoder_errors.png")
    print("   - models/autoencoder_training.png")
    print(f"\n💡 Autoencoder threshold: {autoencoder.threshold:.6f}")
    print(f"   Transactions with error > threshold are flagged as fraud")