from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score,classification_report


def prepare_model_smote(df, target_column, exclude_columns): 
    # Select numeric features and remove excluded columns
    numeric_features = df.select_dtypes(include=np.number).columns
    features = [col for col in numeric_features 
                if col != target_column and col not in exclude_columns]
    
    X = df[features]
    y = df[target_column]
    
    # Split data and balance training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    sm = SMOTE(random_state=0)
    X_train, y_train = sm.fit_resample(X_train, y_train) 
    
    return X_train, X_test, y_train, y_test


def execute_model(X_train, X_test, y_train, y_test):
    # Train decision tree model
    model = DecisionTreeClassifier(random_state=13, criterion='entropy')
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    
    # Extract customers predicted to churn
    churn_customers = X_test[predictions == 1].copy()
    churn_customers['ChurnProbability'] = probabilities[predictions == 1]
    churn_customers['ActualChurn'] = y_test[predictions == 1]
    
    # Evaluate model
    auc_score = roc_auc_score(y_test, predictions)
    print(classification_report(y_test, predictions))
    print(f"The area under the curve is: {auc_score:.2f}")
    
    return model, predictions, churn_customers


