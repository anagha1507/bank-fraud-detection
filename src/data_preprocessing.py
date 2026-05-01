import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

class FraudDataPreprocessor:
    """Generate synthetic bank transaction data and preprocess it for fraud detection."""
    
    def __init__(self, n_samples=100000, fraud_ratio=0.01, random_state=42):
        self.n_samples = n_samples
        self.fraud_ratio = fraud_ratio
        self.random_state = random_state
        np.random.seed(random_state)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def generate_synthetic_data(self):
        """Generate realistic synthetic banking transaction data."""
        
        n_fraud = int(self.n_samples * self.fraud_ratio)
        n_normal = self.n_samples - n_fraud
        
        print(f"Generating {n_normal} normal transactions and {n_fraud} fraudulent transactions...")
        
        # Generate normal transactions
        normal_data = self._generate_transactions(n_normal, is_fraud=False)
        
        # Generate fraudulent transactions
        fraud_data = self._generate_transactions(n_fraud, is_fraud=True)
        
        # Combine and shuffle
        df = pd.concat([normal_data, fraud_data], ignore_index=True)
        df = df.sample(frac=1, random_state=self.random_state).reset_index(drop=True)
        
        print(f"Dataset shape: {df.shape}")
        print(f"Fraud percentage: {df['is_fraud'].mean()*100:.2f}%")
        
        return df
    
    def _generate_transactions(self, n, is_fraud=False):
        """Generate transaction records."""
        
        # Time features
        time = np.random.randint(0, 86400, n)  # Seconds in a day
        hour = time // 3600
        day_of_week = np.random.randint(0, 7, n)
        
        # Transaction types
        transaction_types = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
        
        # Normal: balanced distribution | Fraud: heavy on TRANSFER and CASH_OUT
        if is_fraud:
            type_probs = [0.05, 0.30, 0.05, 0.10, 0.50]
        else:
            type_probs = [0.15, 0.20, 0.25, 0.25, 0.15]
        
        txn_type = np.random.choice(transaction_types, n, p=type_probs)
        
        # Amount - Fraud tends to be larger
        if is_fraud:
            amount = np.random.lognormal(mean=6.5, sigma=1.5, size=n)
        else:
            amount = np.random.lognormal(mean=4.0, sigma=1.2, size=n)
        amount = np.clip(amount, 1, 50000).astype(float)
        
        # Account balances
        old_balance_orig = np.random.lognormal(mean=8.0, sigma=1.0, size=n)
        new_balance_orig = old_balance_orig - amount
        
        old_balance_dest = np.random.lognormal(mean=8.0, sigma=1.0, size=n)
        new_balance_dest = old_balance_dest + amount
        
        # Fraud patterns: Destination balance doesn't change properly
        if is_fraud:
            mask = np.random.random(n) > 0.5
            new_balance_dest[mask] = old_balance_dest[mask]  # Money "disappeared"
            new_balance_orig = old_balance_orig.copy()  # Balance didn't update
        
        # Error balances
        error_balance_orig = old_balance_orig - new_balance_orig
        error_balance_dest = old_balance_dest - new_balance_dest
        
        # Customer age (18-90)
        if is_fraud:
            # Fraud targets older customers more
            age = np.random.normal(60, 15, n).astype(int)
        else:
            age = np.random.normal(40, 15, n).astype(int)
        age = np.clip(age, 18, 90)
        
        # Gender
        gender = np.random.choice(['M', 'F'], n)
        
        # Transaction frequency features
        txn_count_24h = np.random.poisson(3, n)
        if is_fraud:
            txn_count_24h = np.random.poisson(8, n)  # Unusual spike
        
        # Device type
        devices = ['Mobile', 'Web', 'ATM', 'POS']
        if is_fraud:
            device_probs = [0.45, 0.35, 0.10, 0.10]
        else:
            device_probs = [0.30, 0.30, 0.20, 0.20]
        device = np.random.choice(devices, n, p=device_probs)
        
        # Location (simplified - distance from home branch in km)
        if is_fraud:
            location_distance = np.random.exponential(500, n)  # Far from home
        else:
            location_distance = np.random.exponential(10, n)
        
        # Previous fraud history
        if is_fraud:
            previous_frauds = np.random.poisson(2, n)
        else:
            previous_frauds = np.random.poisson(0.05, n)
        previous_frauds = np.clip(previous_frauds, 0, 10)
        
        # Account age in days
        if is_fraud:
            account_age = np.random.exponential(30, n)  # New accounts
        else:
            account_age = np.random.exponential(500, n)
        
        # Create DataFrame
        data = {
            'step': np.random.randint(1, 744, n),  # Hours in a month
            'type': txn_type,
            'amount': amount.round(2),
            'oldbalanceOrg': old_balance_orig.round(2),
            'newbalanceOrig': new_balance_orig.round(2),
            'oldbalanceDest': old_balance_dest.round(2),
            'newbalanceDest': new_balance_dest.round(2),
            'errorBalanceOrig': error_balance_orig.round(2),
            'errorBalanceDest': error_balance_dest.round(2),
            'hour': hour,
            'day_of_week': day_of_week,
            'age': age,
            'gender': gender,
            'txn_count_24h': txn_count_24h,
            'device': device,
            'location_distance': location_distance.round(2),
            'previous_frauds': previous_frauds,
            'account_age_days': account_age.round(0),
            'is_fraud': np.ones(n) if is_fraud else np.zeros(n)
        }
        
        return pd.DataFrame(data)
    
    def engineer_features(self, df):
        """Create additional features for fraud detection."""
        print("\nEngineering additional features...")
        
        df = df.copy()
        
        # Balance change features
        df['balance_change_orig'] = df['newbalanceOrig'] - df['oldbalanceOrg']
        df['balance_change_dest'] = df['newbalanceDest'] - df['oldbalanceDest']
        
        # Ratio features
        df['amount_to_old_balance'] = df['amount'] / (df['oldbalanceOrg'] + 1)
        df['amount_to_new_balance'] = df['amount'] / (df['newbalanceOrig'] + 1)
        df['balance_ratio'] = df['oldbalanceOrg'] / (df['oldbalanceDest'] + 1)
        
        # Time-based features
        df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Amount categories
        df['amount_log'] = np.log1p(df['amount'])
        df['high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
        
        # Risk score (simple heuristic)
        df['risk_score'] = (
            (df['high_amount'] * 0.3) +
            (df['is_night'] * 0.1) +
            (df['is_weekend'] * 0.05) +
            ((df['txn_count_24h'] > 10).astype(int) * 0.2) +
            ((df['location_distance'] > 100).astype(int) * 0.2) +
            ((df['previous_frauds'] > 0).astype(int) * 0.15)
        )
        
        print(f"Features after engineering: {df.shape[1]}")
        return df
    
    def preprocess(self, df):
        """Full preprocessing pipeline."""
        print("\n" + "="*50)
        print("PREPROCESSING PIPELINE")
        print("="*50)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Identify column types
        categorical_cols = ['type', 'gender', 'device']
        numerical_cols = [col for col in df.columns 
                         if col not in categorical_cols + ['is_fraud'] 
                         and df[col].dtype in ['float64', 'int64', 'int32']]
        
        # Encode categorical variables
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            df[col] = self.label_encoders[col].fit_transform(df[col])
        
        # Separate features and target
        X = df.drop('is_fraud', axis=1)
        y = df['is_fraud'].astype(int)
        
        # Train-test split (stratified to maintain fraud ratio)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        # Scale numerical features
        X_train[numerical_cols] = self.scaler.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = self.scaler.transform(X_test[numerical_cols])
        
        print(f"Training set: {X_train.shape}, Fraud cases: {y_train.sum()}")
        print(f"Test set: {X_test.shape}, Fraud cases: {y_test.sum()}")
        
        return X_train, X_test, y_train, y_test, numerical_cols, categorical_cols
    
    def apply_smote(self, X_train, y_train):
        """Apply SMOTE to handle class imbalance."""
        print("\nApplying SMOTE for class balancing...")
        smote = SMOTE(sampling_strategy='auto', random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE - Fraud cases: {y_resampled.sum()}, Normal: {len(y_resampled) - y_resampled.sum()}")
        return X_resampled, y_resampled


# Test the preprocessor
if __name__ == "__main__":
    preprocessor = FraudDataPreprocessor(n_samples=50000, fraud_ratio=0.02)
    df = preprocessor.generate_synthetic_data()
    X_train, X_test, y_train, y_test, num_cols, cat_cols = preprocessor.preprocess(df)
    
    # Save processed data
    df.to_csv('../data/raw/raw_transactions.csv', index=False)
    print("\nData saved to data/raw/raw_transactions.csv")
    
    print("\nClass distribution in training set:")
    print(y_train.value_counts())
    print(f"\nFraud rate: {y_train.mean()*100:.2f}%")