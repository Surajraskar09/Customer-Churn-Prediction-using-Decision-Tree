"""
Customer Churn Prediction using Decision Tree
Complete pipeline for telecom customer churn analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from datetime import datetime
import joblib

# ML Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           accuracy_score, precision_score, recall_score, f1_score)
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore')
plt.style.use('default')

class CustomerChurnPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.target_encoder = None
        self.categorical_encoders = {}
        self.churn_insights = {}
        self.df = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_params = None
        self.best_cv_score = None
        self.comparison_models = {}
        self.evaluation_results = {}
        
    def generate_sample_data(self, n_samples=5000, save_csv=True):
        """Generate synthetic telecom customer data"""
        print("=== GENERATING SAMPLE TELECOM DATA ===")
        
        np.random.seed(42)
        
        # Generate customer data
        data = {
            'CustomerID': [f'CUST_{i:06d}' for i in range(1, n_samples + 1)],
            'Gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.52, 0.48]),
            'Age': np.random.normal(45, 15, n_samples).clip(18, 80).astype(int),
            'Tenure': np.random.exponential(24, n_samples).clip(1, 72).astype(int),
            'Contract': np.random.choice(['Month-to-Month', 'One year', 'Two year'], 
                                       n_samples, p=[0.55, 0.25, 0.20]),
            'MonthlyCharges': np.random.normal(65, 25, n_samples).clip(20, 120).round(2),
            'TotalCharges': np.zeros(n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], 
                                              n_samples, p=[0.35, 0.45, 0.20]),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], 
                                             n_samples, p=[0.30, 0.50, 0.20]),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], 
                                          n_samples, p=[0.25, 0.55, 0.20]),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], 
                                          n_samples, p=[0.40, 0.40, 0.20]),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.60, 0.40]),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 
                                             'Bank transfer (automatic)', 'Credit card (automatic)'], 
                                            n_samples, p=[0.35, 0.15, 0.25, 0.25]),
            'SeniorCitizen': np.zeros(n_samples)
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Calculate derived fields
        df['SeniorCitizen'] = (df['Age'] >= 65).astype(int)
        df['TotalCharges'] = (df['MonthlyCharges'] * df['Tenure'] + 
                             np.random.normal(0, 100, n_samples)).clip(0).round(2)
        
        # Generate complaints based on service quality
        complaint_prob = np.where(df['TechSupport'] == 'No', 0.4, 0.1)
        complaint_prob = np.where(df['OnlineSecurity'] == 'No', complaint_prob + 0.2, complaint_prob)
        df['NumComplaints'] = np.random.poisson(complaint_prob * 3).clip(0, 10)
        
        # Generate churn based on realistic factors
        churn_prob = 0.1  # Base churn rate
        
        # Contract type influence
        churn_prob = np.where(df['Contract'] == 'Month-to-Month', 0.35, churn_prob)
        churn_prob = np.where(df['Contract'] == 'One year', 0.15, churn_prob)
        churn_prob = np.where(df['Contract'] == 'Two year', 0.05, churn_prob)
        
        # Monthly charges influence
        churn_prob = np.where(df['MonthlyCharges'] > 80, churn_prob + 0.15, churn_prob)
        churn_prob = np.where(df['MonthlyCharges'] < 30, churn_prob + 0.10, churn_prob)
        
        # Tenure influence (longer tenure = lower churn)
        churn_prob = np.where(df['Tenure'] < 6, churn_prob + 0.25, churn_prob)
        churn_prob = np.where(df['Tenure'] > 36, churn_prob - 0.15, churn_prob)
        
        # Service quality influence
        churn_prob = np.where(df['NumComplaints'] > 3, churn_prob + 0.30, churn_prob)
        churn_prob = np.where(df['TechSupport'] == 'No', churn_prob + 0.10, churn_prob)
        churn_prob = np.where(df['OnlineSecurity'] == 'No', churn_prob + 0.08, churn_prob)
        
        # Payment method influence
        churn_prob = np.where(df['PaymentMethod'] == 'Electronic check', churn_prob + 0.15, churn_prob)
        
        # Senior citizen influence
        churn_prob = np.where(df['SeniorCitizen'] == 1, churn_prob + 0.10, churn_prob)
        
        # Clip probabilities
        churn_prob = np.clip(churn_prob, 0.02, 0.8)
        
        # Generate churn
        df['Churn'] = np.random.binomial(1, churn_prob)
        df['Churn'] = df['Churn'].map({0: 'No', 1: 'Yes'})
        
        if save_csv:
            df.to_csv('telecom_customer_churn.csv', index=False)
            print(f"✅ Sample data saved to 'telecom_customer_churn.csv'")
        
        self.df = df
        print(f"✅ Generated {n_samples:,} customer records")
        print(f"   Churn Rate: {(df['Churn'] == 'Yes').mean()*100:.2f}%")
        
        return df
    
    def load_data(self, file_path=None):
        """Load customer churn data"""
        print("=== LOADING CUSTOMER DATA ===")
        
        if file_path is None:
            # Generate sample data if no file provided
            return self.generate_sample_data()
        
        try:
            self.df = pd.read_csv(file_path)
            print(f"✅ Data loaded successfully!")
            print(f"   Shape: {self.df.shape}")
            print(f"   Columns: {list(self.df.columns)}")
            
            # Basic statistics
            churn_rate = (self.df['Churn'] == 'Yes').mean() * 100
            print(f"   Total Customers: {len(self.df):,}")
            print(f"   Churn Rate: {churn_rate:.2f}%")
            print(f"   Missing Values: {self.df.isnull().sum().sum()}")
            
            return self.df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            print("📝 Generating sample data instead...")
            return self.generate_sample_data()
    
    def data_preprocessing(self):
        """Clean and preprocess the data"""
        print("\n=== DATA PREPROCESSING ===")
        
        # Handle missing values
        print(f"Missing values before cleaning: {self.df.isnull().sum().sum()}")
        
        # Convert TotalCharges to numeric (some datasets have ' ' for missing)
        if self.df['TotalCharges'].dtype == 'object':
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges'], errors='coerce')
            self.df['TotalCharges'].fillna(self.df['TotalCharges'].median(), inplace=True)
        
        # Create additional features
        self.df['ChargesPerMonth'] = self.df['TotalCharges'] / (self.df['Tenure'] + 1)
        self.df['ChargesRatio'] = self.df['MonthlyCharges'] / (self.df['ChargesPerMonth'] + 0.01)
        self.df['HighCharges'] = (self.df['MonthlyCharges'] > self.df['MonthlyCharges'].quantile(0.75)).astype(int)
        self.df['LongTenure'] = (self.df['Tenure'] > 24).astype(int)
        self.df['HighComplaints'] = (self.df['NumComplaints'] > 2).astype(int)
        
        # Handle categorical variables
        categorical_columns = ['Gender', 'Contract', 'InternetService', 'OnlineSecurity',
                              'TechSupport', 'StreamingTV', 'PaperlessBilling', 'PaymentMethod']
        
        # Label encoding for categorical variables
        for col in categorical_columns:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.categorical_encoders[col] = le
        
        # Encode target variable
        self.target_encoder = LabelEncoder()
        self.df['Churn_encoded'] = self.target_encoder.fit_transform(self.df['Churn'])
        
        print(f"✅ Preprocessing completed")
        print(f"   New features created: ChargesPerMonth, ChargesRatio, HighCharges, LongTenure, HighComplaints")
        print(f"   Categorical variables encoded: {len(categorical_columns)}")
        
    def exploratory_data_analysis(self):
        """Comprehensive EDA"""
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        
        # Create comprehensive EDA plots
        fig, axes = plt.subplots(4, 3, figsize=(20, 20))
        fig.suptitle('Customer Churn Analysis - Exploratory Data Analysis', fontsize=16, fontweight='bold')
        
        # 1. Churn Distribution
        churn_counts = self.df['Churn'].value_counts()
        axes[0,0].pie(churn_counts.values, labels=churn_counts.index, autopct='%1.1f%%', 
                     colors=['lightblue', 'salmon'])
        axes[0,0].set_title('Overall Churn Distribution')
        
        # 2. Churn by Contract Type
        contract_churn = pd.crosstab(self.df['Contract'], self.df['Churn'], normalize='index') * 100
        contract_churn.plot(kind='bar', ax=axes[0,1], color=['lightblue', 'salmon'])
        axes[0,1].set_title('Churn Rate by Contract Type')
        axes[0,1].set_ylabel('Churn Rate (%)')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].legend(title='Churn')
        
        # 3. Monthly Charges Distribution
        churned = self.df[self.df['Churn'] == 'Yes']['MonthlyCharges']
        not_churned = self.df[self.df['Churn'] == 'No']['MonthlyCharges']
        axes[0,2].hist(not_churned, bins=30, alpha=0.7, label='No Churn', color='lightblue', density=True)
        axes[0,2].hist(churned, bins=30, alpha=0.7, label='Churned', color='salmon', density=True)
        axes[0,2].set_xlabel('Monthly Charges ($)')
        axes[0,2].set_ylabel('Density')
        axes[0,2].set_title('Monthly Charges Distribution by Churn')
        axes[0,2].legend()
        
        # 4. Tenure vs Churn
        tenure_churn = self.df.groupby('Tenure')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
        axes[1,0].plot(tenure_churn.index, tenure_churn.values, marker='o', color='red')
        axes[1,0].set_xlabel('Tenure (months)')
        axes[1,0].set_ylabel('Churn Rate (%)')
        axes[1,0].set_title('Churn Rate vs Tenure')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Age Distribution
        age_churned = self.df[self.df['Churn'] == 'Yes']['Age']
        age_not_churned = self.df[self.df['Churn'] == 'No']['Age']
        axes[1,1].hist(age_not_churned, bins=25, alpha=0.7, label='No Churn', color='lightblue', density=True)
        axes[1,1].hist(age_churned, bins=25, alpha=0.7, label='Churned', color='salmon', density=True)
        axes[1,1].set_xlabel('Age')
        axes[1,1].set_ylabel('Density')
        axes[1,1].set_title('Age Distribution by Churn')
        axes[1,1].legend()
        
        # 6. Internet Service vs Churn
        internet_churn = pd.crosstab(self.df['InternetService'], self.df['Churn'], normalize='index') * 100
        internet_churn.plot(kind='bar', ax=axes[1,2], color=['lightblue', 'salmon'])
        axes[1,2].set_title('Churn Rate by Internet Service')
        axes[1,2].set_ylabel('Churn Rate (%)')
        axes[1,2].tick_params(axis='x', rotation=45)
        axes[1,2].legend(title='Churn')
        
        # 7. Payment Method vs Churn
        payment_churn = pd.crosstab(self.df['PaymentMethod'], self.df['Churn'], normalize='index') * 100
        payment_churn.plot(kind='bar', ax=axes[2,0], color=['lightblue', 'salmon'])
        axes[2,0].set_title('Churn Rate by Payment Method')
        axes[2,0].set_ylabel('Churn Rate (%)')
        axes[2,0].tick_params(axis='x', rotation=45)
        axes[2,0].legend(title='Churn')
        
        # 8. Complaints vs Churn
        complaint_churn = self.df.groupby('NumComplaints')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)
        axes[2,1].bar(complaint_churn.index, complaint_churn.values, color='coral', alpha=0.7)
        axes[2,1].set_xlabel('Number of Complaints')
        axes[2,1].set_ylabel('Churn Rate (%)')
        axes[2,1].set_title('Churn Rate by Number of Complaints')
        
        # 9. Correlation Matrix
        numerical_cols = ['Age', 'Tenure', 'MonthlyCharges', 'TotalCharges', 
                         'NumComplaints', 'SeniorCitizen', 'Churn_encoded']
        corr_matrix = self.df[numerical_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, fmt='.2f', ax=axes[2,2])
        axes[2,2].set_title('Feature Correlation Matrix')
        
        # 10. Senior Citizens Churn
        senior_churn = pd.crosstab(self.df['SeniorCitizen'], self.df['Churn'], normalize='index') * 100
        senior_churn.plot(kind='bar', ax=axes[3,0], color=['lightblue', 'salmon'])
        axes[3,0].set_title('Churn Rate: Senior vs Non-Senior')
        axes[3,0].set_xlabel('Senior Citizen (0=No, 1=Yes)')
        axes[3,0].set_ylabel('Churn Rate (%)')
        axes[3,0].tick_params(axis='x', rotation=0)
        axes[3,0].legend(title='Churn')
        
        # 11. Tech Support vs Churn
        tech_churn = pd.crosstab(self.df['TechSupport'], self.df['Churn'], normalize='index') * 100
        tech_churn.plot(kind='bar', ax=axes[3,1], color=['lightblue', 'salmon'])
        axes[3,1].set_title('Churn Rate by Tech Support')
        axes[3,1].set_ylabel('Churn Rate (%)')
        axes[3,1].tick_params(axis='x', rotation=45)
        axes[3,1].legend(title='Churn')
        
        # 12. Total Charges vs Monthly Charges (Scatter)
        churned_scatter = self.df[self.df['Churn'] == 'Yes']
        not_churned_scatter = self.df[self.df['Churn'] == 'No'].sample(min(500, len(self.df[self.df['Churn'] == 'No'])), random_state=42)
        
        axes[3,2].scatter(not_churned_scatter['MonthlyCharges'], not_churned_scatter['TotalCharges'], 
                         alpha=0.5, label='No Churn', color='lightblue', s=20)
        axes[3,2].scatter(churned_scatter['MonthlyCharges'], churned_scatter['TotalCharges'], 
                         alpha=0.7, label='Churned', color='salmon', s=20)
        axes[3,2].set_xlabel('Monthly Charges ($)')
        axes[3,2].set_ylabel('Total Charges ($)')
        axes[3,2].set_title('Total vs Monthly Charges by Churn')
        axes[3,2].legend()
        
        plt.tight_layout()
        plt.show()
        
        # Store insights
        self.churn_insights = {
            'overall_churn_rate': (self.df['Churn'] == 'Yes').mean() * 100,
            'high_risk_contract': contract_churn['Yes'].idxmax(),
            'high_risk_payment': payment_churn['Yes'].idxmax(),
            'avg_complaints_churned': self.df[self.df['Churn'] == 'Yes']['NumComplaints'].mean(),
            'avg_complaints_retained': self.df[self.df['Churn'] == 'No']['NumComplaints'].mean()
        }
        
        print("📊 EDA completed! Key insights stored.")
    
    def prepare_features(self):
        """Prepare features for model training"""
        print("\n=== FEATURE PREPARATION ===")
        
        # Select features for model training
        feature_columns = [
            'Age', 'Tenure', 'MonthlyCharges', 'TotalCharges', 'ChargesPerMonth',
            'NumComplaints', 'SeniorCitizen', 'HighCharges', 'LongTenure', 'HighComplaints',
            'Gender_encoded', 'Contract_encoded', 'InternetService_encoded',
            'OnlineSecurity_encoded', 'TechSupport_encoded', 'StreamingTV_encoded',
            'PaperlessBilling_encoded', 'PaymentMethod_encoded'
        ]
        
        # Filter existing columns
        available_features = [col for col in feature_columns if col in self.df.columns]
        
        self.X = self.df[available_features]
        self.y = self.df['Churn_encoded']
        self.feature_names = available_features
        
        print(f"✅ Features prepared: {len(available_features)} columns")
        print(f"   Feature columns: {available_features}")
        print(f"   Target distribution: {self.df['Churn'].value_counts().to_dict()}")
        
        return self.X, self.y
    
    def train_test_split_data(self, test_size=0.2, random_state=42):
        """Split data for training and testing"""
        print("\n=== TRAIN-TEST SPLIT ===")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state, stratify=self.y
        )
        
        print(f"✅ Data split completed:")
        print(f"   Training set: {self.X_train.shape[0]:,} samples")
        print(f"   Test set: {self.X_test.shape[0]:,} samples")
        print(f"   Training churn rate: {self.y_train.mean()*100:.2f}%")
        print(f"   Test churn rate: {self.y_test.mean()*100:.2f}%")
    
    def train_decision_tree(self, tune_hyperparameters=True):
        """Train Decision Tree with hyperparameter tuning"""
        print("\n=== TRAINING DECISION TREE ===")
        
        if tune_hyperparameters:
            print("🔄 Performing hyperparameter tuning...")
            
            # Define parameter grid
            param_grid = {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10, 20],
                'min_samples_leaf': [1, 2, 5, 10],
                'max_features': ['sqrt', 'log2', None],
                'criterion': ['gini', 'entropy']
            }
            
            # Grid search with cross-validation
            dt_base = DecisionTreeClassifier(random_state=42)
            grid_search = GridSearchCV(
                dt_base, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=0
            )
            
            grid_search.fit(self.X_train, self.y_train)
            
            self.model = grid_search.best_estimator_
            self.best_params = grid_search.best_params_
            self.best_cv_score = grid_search.best_score_
            
            print(f"✅ Best parameters found: {self.best_params}")
            print(f"✅ Best CV F1-score: {self.best_cv_score:.4f}")
        
        else:
            print("🔄 Training with default parameters...")
            self.model = DecisionTreeClassifier(
                max_depth=7, min_samples_split=10, min_samples_leaf=5,
                random_state=42
            )
            self.model.fit(self.X_train, self.y_train)
            print("✅ Model trained with default parameters")
        
        # Train additional models for comparison
        self.comparison_models = {}
        
        # Random Forest for comparison
        print("🔄 Training Random Forest for comparison...")
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(self.X_train, self.y_train)
        self.comparison_models['Random Forest'] = rf_model
        
        print("✅ All models trained successfully!")
    
    def evaluate_model(self):
        """Comprehensive model evaluation"""
        print("\n=== MODEL EVALUATION ===")
        
        # Get predictions for all models
        models_to_evaluate = {'Decision Tree': self.model}
        models_to_evaluate.update(self.comparison_models)
        
        self.evaluation_results = {}
        
        # Create evaluation plots
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Customer Churn Prediction - Model Evaluation', fontsize=16, fontweight='bold')
        
        colors = ['blue', 'red', 'green']
        
        for idx, (name, model) in enumerate(models_to_evaluate.items()):
            # Predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            
            # Store results
            self.evaluation_results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            if y_pred_proba is not None:
                roc_auc = roc_auc_score(self.y_test, y_pred_proba)
                self.evaluation_results[name]['roc_auc'] = roc_auc
            
            print(f"\n📊 {name} Results:")
            print(f"   Accuracy:  {accuracy:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"   F1-Score:  {f1:.4f}")
            if y_pred_proba is not None:
                print(f"   ROC-AUC:   {roc_auc:.4f}")
            
            # Confusion Matrix (only for first two models due to space)
            if idx < 2:
                cm = confusion_matrix(self.y_test, y_pred)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, idx])
                axes[0, idx].set_title(f'{name} - Confusion Matrix')
                axes[0, idx].set_xlabel('Predicted')
                axes[0, idx].set_ylabel('Actual')
            
            # ROC Curve (only for models with probability prediction)
            if y_pred_proba is not None and idx < 2:
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
                axes[1, idx].plot(fpr, tpr, color=colors[idx], 
                                 label=f'{name} (AUC = {roc_auc:.3f})')
                axes[1, idx].plot([0, 1], [0, 1], 'k--', alpha=0.5)
                axes[1, idx].set_xlabel('False Positive Rate')
                axes[1, idx].set_ylabel('True Positive Rate')
                axes[1, idx].set_title(f'{name} - ROC Curve')
                axes[1, idx].legend()
                axes[1, idx].grid(True)
        
        # Model Comparison
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        comparison_df = pd.DataFrame({
            name: [results.get(metric, 0) for metric in metrics]
            for name, results in self.evaluation_results.items()
        }, index=metrics)
        
        comparison_df.plot(kind='bar', ax=axes[0, 2])
        axes[0, 2].set_title('Model Comparison')
        axes[0, 2].set_xlabel('Metrics')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].legend()
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Feature Importance (Decision Tree)
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(10)
            
            axes[1, 2].barh(range(len(feature_importance)), feature_importance['importance'])
            axes[1, 2].set_yticks(range(len(feature_importance)))
            axes[1, 2].set_yticklabels(feature_importance['feature'])
            axes[1, 2].set_xlabel('Feature Importance')
            axes[1, 2].set_title('Top 10 Feature Importance (Decision Tree)')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return comparison_df
    
    def visualize_decision_tree(self, max_depth=3):
        """Visualize the decision tree"""
        print("\n=== DECISION TREE VISUALIZATION ===")
        
        # Create a simpler tree for visualization
        simple_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        simple_tree.fit(self.X_train, self.y_train)
        
        # Plot the tree
        plt.figure(figsize=(20, 10))
        plot_tree(simple_tree, 
                 feature_names=self.feature_names,
                 class_names=['No Churn', 'Churn'],
                 filled=True,
                 rounded=True,
                 fontsize=10)
        plt.title(f'Decision Tree Visualization (max_depth={max_depth})')
        plt.show()
        
        # Print text representation
        tree_rules = export_text(simple_tree, feature_names=self.feature_names)
        print("\n📋 Decision Tree Rules (simplified):")
        print("=" * 50)
        print(tree_rules[:2000] + "..." if len(tree_rules) > 2000 else tree_rules)
        
        print("✅ Decision tree visualization completed!")
    
    def predict_churn(self, customer_data):
        """Predict churn for new customer data"""
        if self.model is None:
            print("❌ Model not trained yet!")
            return None
        
        try:
            # If customer_data is a dictionary, convert to DataFrame
            if isinstance(customer_data, dict):
                customer_df = pd.DataFrame([customer_data])
            else:
                customer_df = customer_data.copy()
            
            # Apply same preprocessing as training data
            for col, encoder in self.categorical_encoders.items():
                if col in customer_df.columns:
                    try:
                        customer_df[f'{col}_encoded'] = encoder.transform(customer_df[col].astype(str))
                    except ValueError:
                        # Handle unseen categories
                        customer_df[f'{col}_encoded'] = 0
            
            # Create derived features
            customer_df['ChargesPerMonth'] = customer_df['TotalCharges'] / (customer_df['Tenure'] + 1)
            customer_df['ChargesRatio'] = customer_df['MonthlyCharges'] / (customer_df['ChargesPerMonth'] + 0.01)
            customer_df['HighCharges'] = (customer_df['MonthlyCharges'] > 75).astype(int)  # Using approximate threshold
            customer_df['LongTenure'] = (customer_df['Tenure'] > 24).astype(int)
            customer_df['HighComplaints'] = (customer_df['NumComplaints'] > 2).astype(int)
            
            # Select only the features used in training
            customer_features = customer_df[self.feature_names]
            
            # Make prediction
            prediction = self.model.predict(customer_features)
            prediction_proba = self.model.predict_proba(customer_features)
            
            # Convert back to original labels
            prediction_label = self.target_encoder.inverse_transform(prediction)
            
            result = {
                'prediction': prediction_label[0],
                'churn_probability': prediction_proba[0][1],
                'confidence': max(prediction_proba[0]) * 100
            }
            
            return result
            
        except Exception as e:
            print(f"❌ Error in prediction: {e}")
            return None
    
    def save_model(self, filename='churn_prediction_model.pkl'):
        """Save the trained model and encoders"""
        if self.model is None:
            print("❌ No model to save!")
            return False
        
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'target_encoder': self.target_encoder,
            'categorical_encoders': self.categorical_encoders,
            'best_params': self.best_params,
            'evaluation_results': self.evaluation_results
        }
        
        joblib.dump(model_data, filename)
        print(f"✅ Model saved successfully to {filename}")
        return True
    
    def load_model(self, filename='churn_prediction_model.pkl'):
        """Load a trained model and encoders"""
        try:
            model_data = joblib.load(filename)
            self.model = model_data['model']
            self.feature_names = model_data['feature_names']
            self.target_encoder = model_data['target_encoder']
            self.categorical_encoders = model_data['categorical_encoders']
            self.best_params = model_data.get('best_params', {})
            self.evaluation_results = model_data.get('evaluation_results', {})
            print(f"✅ Model loaded successfully from {filename}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def generate_insights_report(self):
        """Generate a comprehensive insights report"""
        print("\n" + "="*60)
        print("             CUSTOMER CHURN INSIGHTS REPORT")
        print("="*60)
        
        if not self.churn_insights:
            print("❌ No insights available. Run EDA first!")
            return
        
        print(f"\n📊 OVERALL CHURN STATISTICS:")
        print(f"   • Overall Churn Rate: {self.churn_insights['overall_churn_rate']:.2f}%")
        print(f"   • Highest Risk Contract: {self.churn_insights['high_risk_contract']}")
        print(f"   • Highest Risk Payment Method: {self.churn_insights['high_risk_payment']}")
        
        print(f"\n📞 CUSTOMER SERVICE INSIGHTS:")
        print(f"   • Average Complaints (Churned): {self.churn_insights['avg_complaints_churned']:.2f}")
        print(f"   • Average Complaints (Retained): {self.churn_insights['avg_complaints_retained']:.2f}")
        
        if self.evaluation_results:
            print(f"\n🎯 MODEL PERFORMANCE:")
            for model_name, results in self.evaluation_results.items():
                print(f"   • {model_name}:")
                print(f"     - Accuracy: {results['accuracy']:.3f}")
                print(f"     - Precision: {results['precision']:.3f}")
                print(f"     - Recall: {results['recall']:.3f}")
                print(f"     - F1-Score: {results['f1']:.3f}")
                if 'roc_auc' in results:
                    print(f"     - ROC-AUC: {results['roc_auc']:.3f}")
        
        if hasattr(self, 'model') and self.model and hasattr(self.model, 'feature_importances_'):
            print(f"\n🔍 TOP 5 MOST IMPORTANT FEATURES:")
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False).head(5)
            
            for idx, row in feature_importance.iterrows():
                print(f"   • {row['feature']}: {row['importance']:.3f}")
        
        print(f"\n💡 BUSINESS RECOMMENDATIONS:")
        print(f"   • Focus retention efforts on month-to-month customers")
        print(f"   • Improve customer service to reduce complaints")
        print(f"   • Offer incentives for longer-term contracts")
        print(f"   • Monitor high-charge customers more closely")
        print(f"   • Provide better tech support and online security")
        
        print("="*60)

# Main execution function
def run_complete_analysis():
    """Run the complete churn prediction analysis pipeline"""
    print("🚀 Starting Complete Customer Churn Prediction Analysis")
    print("="*70)
    
    # Initialize predictor
    predictor = CustomerChurnPredictor()
    
    try:
        # Step 1: Load or generate data
        predictor.load_data()
        
        # Step 2: Preprocess data
        predictor.data_preprocessing()
        
        # Step 3: Exploratory Data Analysis
        predictor.exploratory_data_analysis()
        
        # Step 4: Prepare features
        predictor.prepare_features()
        
        # Step 5: Split data
        predictor.train_test_split_data()
        
        # Step 6: Train model
        predictor.train_decision_tree(tune_hyperparameters=True)
        
        # Step 7: Evaluate model
        comparison_results = predictor.evaluate_model()
        print(f"\n📋 Model Comparison Results:")
        print(comparison_results)
        
        # Step 8: Visualize decision tree
        predictor.visualize_decision_tree(max_depth=3)
        
        # Step 9: Generate insights report
        predictor.generate_insights_report()
        
        # Step 10: Save model
        predictor.save_model()
        
        # Step 11: Test prediction on sample customer
        print(f"\n🧪 TESTING PREDICTION ON SAMPLE CUSTOMER:")
        sample_customer = {
            'Age': 45,
            'Tenure': 12,
            'MonthlyCharges': 85.50,
            'TotalCharges': 1200.00,
            'NumComplaints': 3,
            'SeniorCitizen': 0,
            'Gender': 'Male',
            'Contract': 'Month-to-Month',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'Yes',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check'
        }
        
        prediction_result = predictor.predict_churn(sample_customer)
        if prediction_result:
            print(f"   Customer Profile: {sample_customer}")
            print(f"   Prediction: {prediction_result['prediction']}")
            print(f"   Churn Probability: {prediction_result['churn_probability']:.3f}")
            print(f"   Confidence: {prediction_result['confidence']:.1f}%")
        
        print(f"\n🎉 Analysis completed successfully!")
        print("="*70)
        
        return predictor
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

# Example usage and testing
def demo_usage():
    """Demonstrate how to use the CustomerChurnPredictor"""
    print("📚 DEMO: How to use the CustomerChurnPredictor")
    print("="*50)
    
    # Create predictor instance
    predictor = CustomerChurnPredictor()
    
    # Generate sample data
    predictor.generate_sample_data(n_samples=1000, save_csv=False)
    
    # Quick analysis
    predictor.data_preprocessing()
    predictor.prepare_features()
    predictor.train_test_split_data()
    predictor.train_decision_tree(tune_hyperparameters=False)  # Faster training
    
    # Quick evaluation
    predictor.evaluate_model()
    
    print("✅ Demo completed!")

if __name__ == "__main__":
    # Run complete analysis
    predictor = run_complete_analysis()
    
    # If you want to run just a quick demo, uncomment the line below:
    # demo_usage()