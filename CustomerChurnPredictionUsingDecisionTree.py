"""
Customer Churn Prediction using Decision Tree - Compact Version
Streamlined pipeline for telecom customer churn analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import warnings
import joblib

warnings.filterwarnings('ignore')

class ChurnPredictor:
    def __init__(self):
        self.model = None
        self.encoders = {}
        self.features = None
        
    def create_sample_data(self, n_samples=3000):
        """Generate realistic telecom customer data"""
        print("=== GENERATING SAMPLE DATA ===")
        np.random.seed(42)
        
        # Generate base customer data
        data = {
            'CustomerID': [f'CUST_{i:05d}' for i in range(n_samples)],
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Age': np.random.normal(42, 15, n_samples).clip(18, 80).astype(int),
            'Tenure': np.random.exponential(24, n_samples).clip(1, 72).astype(int),
            'Contract': np.random.choice(['Month-to-Month', 'One year', 'Two year'], 
                                       n_samples, p=[0.55, 0.25, 0.20]),
            'MonthlyCharges': np.random.normal(65, 20, n_samples).clip(20, 120),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                              n_samples, p=[0.35, 0.45, 0.20]),
            'TechSupport': np.random.choice(['Yes', 'No'], n_samples, p=[0.30, 0.70]),
            'NumComplaints': np.random.poisson(0.8, n_samples).clip(0, 10)
        }
        
        df = pd.DataFrame(data)
        df['TotalCharges'] = df['MonthlyCharges'] * df['Tenure']
        
        # Generate realistic churn based on key factors
        churn_prob = 0.1  # Base rate
        
        # Contract influence (biggest factor)
        churn_prob = np.where(df['Contract'] == 'Month-to-Month', 0.35, 
                             np.where(df['Contract'] == 'One year', 0.15, 0.05))
        
        # Other influences
        churn_prob += np.where(df['MonthlyCharges'] > 80, 0.15, 0)
        churn_prob += np.where(df['Tenure'] < 6, 0.20, 0)
        churn_prob += np.where(df['NumComplaints'] > 2, 0.25, 0)
        churn_prob += np.where(df['TechSupport'] == 'No', 0.10, 0)
        
        churn_prob = np.clip(churn_prob, 0.02, 0.8)
        df['Churn'] = np.random.binomial(1, churn_prob)
        df['Churn'] = df['Churn'].map({0: 'No', 1: 'Yes'})
        
        print(f"✅ Generated {n_samples} customers")
        print(f"   Churn Rate: {(df['Churn'] == 'Yes').mean()*100:.1f}%")
        
        return df
    
    def load_data(self, file_path=None):
        """Load customer data or create sample data"""
        if file_path and pd.io.common.file_exists(file_path):
            df = pd.read_csv(file_path)
            print(f"✅ Loaded data: {df.shape}")
        else:
            print("📝 Creating sample data...")
            df = self.create_sample_data()
            
        return df
    
    def preprocess_data(self, df):
        """Clean and encode the data"""
        print("\n=== DATA PREPROCESSING ===")
        
        # Handle missing values
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
        
        # Create useful features
        df['AvgChargesPerMonth'] = df['TotalCharges'] / (df['Tenure'] + 1)
        df['HighCharges'] = (df['MonthlyCharges'] > df['MonthlyCharges'].quantile(0.75)).astype(int)
        df['NewCustomer'] = (df['Tenure'] <= 6).astype(int)
        
        # Encode categorical variables
        categorical_cols = ['Gender', 'Contract', 'InternetService', 'TechSupport']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le
        
        # Encode target
        le_target = LabelEncoder()
        df['Churn_encoded'] = le_target.fit_transform(df['Churn'])
        self.encoders['Churn'] = le_target
        
        print(f"✅ Preprocessing completed")
        return df
    
    def explore_data(self, df):
        """Quick EDA with key visualizations"""
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        
        # 1. Churn Distribution
        df['Churn'].value_counts().plot(kind='pie', ax=axes[0,0], autopct='%1.1f%%')
        axes[0,0].set_title('Churn Distribution')
        
        # 2. Churn by Contract
        contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index') * 100
        contract_churn.plot(kind='bar', ax=axes[0,1])
        axes[0,1].set_title('Churn Rate by Contract Type')
        axes[0,1].set_ylabel('Churn Rate (%)')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Monthly Charges Distribution
        churned = df[df['Churn'] == 'Yes']['MonthlyCharges']
        not_churned = df[df['Churn'] == 'No']['MonthlyCharges']
        axes[0,2].hist(not_churned, bins=20, alpha=0.7, label='No Churn', density=True)
        axes[0,2].hist(churned, bins=20, alpha=0.7, label='Churned', density=True)
        axes[0,2].set_title('Monthly Charges by Churn')
        axes[0,2].legend()
        
        # 4. Tenure vs Churn
        tenure_churn = df.groupby('Tenure')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
        axes[1,0].plot(tenure_churn.index, tenure_churn.values)
        axes[1,0].set_title('Churn Rate vs Tenure')
        axes[1,0].set_xlabel('Tenure (months)')
        axes[1,0].set_ylabel('Churn Rate (%)')
        
        # 5. Complaints vs Churn
        complaint_churn = df.groupby('NumComplaints')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
        axes[1,1].bar(complaint_churn.index, complaint_churn.values)
        axes[1,1].set_title('Churn Rate by Complaints')
        axes[1,1].set_xlabel('Number of Complaints')
        axes[1,1].set_ylabel('Churn Rate (%)')
        
        # 6. Correlation Matrix
        numeric_cols = ['Age', 'Tenure', 'MonthlyCharges', 'NumComplaints', 'Churn_encoded']
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, ax=axes[1,2])
        axes[1,2].set_title('Feature Correlations')
        
        plt.tight_layout()
        plt.show()
        
        # Print key insights
        print(f"📊 Key Insights:")
        print(f"   Overall Churn Rate: {(df['Churn'] == 'Yes').mean()*100:.1f}%")
        print(f"   Highest Risk Contract: {contract_churn['Yes'].idxmax()}")
        print(f"   Avg Monthly Charge (Churned): ${churned.mean():.0f}")
        print(f"   Avg Monthly Charge (Retained): ${not_churned.mean():.0f}")
    
    def prepare_features(self, df):
        """Prepare feature matrix for training"""
        print("\n=== FEATURE PREPARATION ===")
        
        feature_cols = [
            'Age', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'NumComplaints',
            'AvgChargesPerMonth', 'HighCharges', 'NewCustomer',
            'Gender_encoded', 'Contract_encoded', 'InternetService_encoded', 'TechSupport_encoded'
        ]
        
        # Filter existing columns
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features]
        y = df['Churn_encoded']
        self.features = available_features
        
        print(f"✅ Features prepared: {len(available_features)} columns")
        return X, y
    
    def train_model(self, X, y, tune_params=True):
        """Train Decision Tree with optional hyperparameter tuning"""
        print("\n=== MODEL TRAINING ===")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if tune_params:
            print("🔄 Hyperparameter tuning...")
            param_grid = {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [10, 20, 50],
                'min_samples_leaf': [5, 10, 20],
                'criterion': ['gini', 'entropy']
            }
            
            grid_search = GridSearchCV(
                DecisionTreeClassifier(random_state=42), 
                param_grid, cv=5, scoring='f1', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            
            print(f"✅ Best params: {grid_search.best_params_}")
            print(f"✅ Best CV F1-score: {grid_search.best_score_:.3f}")
        else:
            self.model = DecisionTreeClassifier(max_depth=7, min_samples_split=20, random_state=42)
            self.model.fit(X_train, y_train)
            print("✅ Model trained with default parameters")
        
        return X_train, X_test, y_train, y_test
    
    def evaluate_model(self, X_test, y_test):
        """Evaluate the trained model"""
        print("\n=== MODEL EVALUATION ===")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"📊 Model Performance:")
        print(f"   Accuracy: {accuracy:.3f}")
        print(f"   F1-Score: {f1:.3f}")
        
        # Detailed classification report
        print(f"\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Churn', 'Churn'], 
                   yticklabels=['No Churn', 'Churn'])
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()
        
        # Feature Importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df.head(10), x='importance', y='feature')
            plt.title('Top 10 Feature Importance')
            plt.tight_layout()
            plt.show()
            
            print(f"\n🔍 Top 5 Important Features:")
            for _, row in importance_df.head(5).iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
    
    def visualize_tree(self, max_depth=3):
        """Visualize the decision tree"""
        print("\n=== DECISION TREE VISUALIZATION ===")
        
        # Create a simplified tree for visualization
        simple_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        # We need to retrain on the data for visualization
        # This is a simplified version - you may need to pass the training data
        
        plt.figure(figsize=(15, 8))
        plot_tree(self.model if self.model.get_depth() <= max_depth else simple_tree,
                 feature_names=self.features,
                 class_names=['No Churn', 'Churn'],
                 filled=True, rounded=True, fontsize=10)
        plt.title(f'Decision Tree (depth ≤ {max_depth})')
        plt.show()
    
    def predict_single_customer(self, customer_data):
        """Predict churn for a single customer"""
        if self.model is None:
            print("❌ Model not trained!")
            return None
        
        try:
            # Convert to DataFrame if needed
            if isinstance(customer_data, dict):
                customer_df = pd.DataFrame([customer_data])
            else:
                customer_df = customer_data.copy()
            
            # Apply preprocessing
            for col, encoder in self.encoders.items():
                if col in customer_df.columns and col != 'Churn':
                    try:
                        customer_df[f'{col}_encoded'] = encoder.transform(customer_df[col].astype(str))
                    except ValueError:
                        customer_df[f'{col}_encoded'] = 0  # Handle unseen categories
            
            # Create derived features
            customer_df['AvgChargesPerMonth'] = customer_df['TotalCharges'] / (customer_df['Tenure'] + 1)
            customer_df['HighCharges'] = (customer_df['MonthlyCharges'] > 75).astype(int)
            customer_df['NewCustomer'] = (customer_df['Tenure'] <= 6).astype(int)
            
            # Select features
            customer_features = customer_df[self.features]
            
            # Predict
            prediction = self.model.predict(customer_features)[0]
            probability = self.model.predict_proba(customer_features)[0]
            
            result = {
                'prediction': self.encoders['Churn'].inverse_transform([prediction])[0],
                'churn_probability': probability[1],
                'confidence': max(probability) * 100
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None
    
    def save_model(self, filename='churn_model.pkl'):
        """Save the trained model and encoders"""
        model_data = {
            'model': self.model,
            'encoders': self.encoders,
            'features': self.features
        }
        joblib.dump(model_data, filename)
        print(f"✅ Model saved to {filename}")
    
    def load_model(self, filename='churn_model.pkl'):
        """Load a saved model"""
        try:
            model_data = joblib.load(filename)
            self.model = model_data['model']
            self.encoders = model_data['encoders']
            self.features = model_data['features']
            print(f"✅ Model loaded from {filename}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

def main():
    """Run the complete churn prediction pipeline"""
    print("🚀 CUSTOMER CHURN PREDICTION - DECISION TREE")
    print("="*60)
    
    # Initialize predictor
    predictor = ChurnPredictor()
    
    # Step 1: Load Data
    # Change this path to your dataset file
    data_file = None  # Set to "your_churn_dataset.csv" if you have one
    df = predictor.load_data(data_file)
    
    # Step 2: Preprocess Data
    df = predictor.preprocess_data(df)
    
    # Step 3: Exploratory Data Analysis
    predictor.explore_data(df)
    
    # Step 4: Prepare Features
    X, y = predictor.prepare_features(df)
    
    # Step 5: Train Model
    X_train, X_test, y_train, y_test = predictor.train_model(X, y, tune_params=True)
    
    # Step 6: Evaluate Model
    predictor.evaluate_model(X_test, y_test)
    
    # Step 7: Visualize Tree
    predictor.visualize_tree(max_depth=3)
    
    # Step 8: Test Prediction
    print("\n=== TESTING PREDICTION ===")
    sample_customer = {
        'Age': 45,
        'Tenure': 8,
        'MonthlyCharges': 85.0,
        'TotalCharges': 680.0,
        'NumComplaints': 3,
        'Gender': 'Male',
        'Contract': 'Month-to-Month',
        'InternetService': 'Fiber optic',
        'TechSupport': 'No'
    }
    
    result = predictor.predict_single_customer(sample_customer)
    if result:
        print(f"🧪 Sample Customer Prediction:")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Churn Probability: {result['churn_probability']:.3f}")
        print(f"   Confidence: {result['confidence']:.1f}%")
    
    # Step 9: Save Model
    predictor.save_model('churn_decision_tree.pkl')
    
    print("\n✅ Analysis Complete!")
    print("="*60)
    
    return predictor

# Quick demo function
def quick_demo():
    """Run a quick demonstration"""
    predictor = ChurnPredictor()
    df = predictor.create_sample_data(1000)
    df = predictor.preprocess_data(df)
    X, y = predictor.prepare_features(df)
    X_train, X_test, y_train, y_test = predictor.train_model(X, y, tune_params=False)
    predictor.evaluate_model(X_test, y_test)
    print("✅ Quick demo completed!")

if __name__ == "__main__":
    # Run full analysis
    predictor = main()
    
    # Uncomment for quick demo:
    # quick_demo()
