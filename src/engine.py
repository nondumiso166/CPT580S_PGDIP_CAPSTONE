# import the required libraries
import pickle
from ml_pipeline.utils import read_dataset,examine,null_values
from ml_pipeline.model import prepare_model_smote,execute_model
from ml_pipeline.evaluation_metrics import evaluate_classification,plot_roc_curve
from ml_pipeline.feature_importance import plot_perm_importance
from ml_pipeline.plot_model import plot_model
import matplotlib.pyplot as plt
import pandas as pd


# Read the initial dataset
df = pd.read_csv(r"C:\Users\gagan\OneDrive\Desktop\CPT580S_PGDIP_CAPSTONE\src/ecommercedata.csv")


# View first five columns of the dataset

print(df.head())

# We need to examine and clean the data. check for null values, 

x =examine(df)

df = null_values(df)

print("Total rows after dropping nulls: ",df.count().iloc[0])

# print(a.head())



### Run the decision tree model with sklearn ###




# Load and prepare data
print("Preparing data...")
X_train, X_test, y_train, y_test = prepare_model_smote(
    df, 
    'Churn', 
    ['CustomerID']  # Exclude ID from features but keep for identification
)

# Train model and get predictions
print("Training model...")
model_dectree, y_pred, churn_customers = execute_model(X_train, X_test, y_train, y_test)

# 3. Reattach CustomerID to identified churners
print("Identifying churn customers...")
churn_customers_with_ids = df.loc[churn_customers.index].copy()
churn_customers_with_ids['ChurnProbability'] = churn_customers['ChurnProbability']
churn_customers_with_ids['PredictedChurn'] = 1

# Display results
print(f"Found {len(churn_customers_with_ids)} customers at risk of churning")
print("\n Top 10 highest risk customers:")
print(churn_customers_with_ids[['CustomerID', 'ChurnProbability']].head(10))

#  Save results
churn_customers_with_ids.to_csv('customers_at_risk.csv', index=False)
print("Results saved to 'customers_at_risk.csv'")



## performance metric ##
conf_matrix = evaluate_classification(y_test,y_pred) # generate confusion matrix
#print(conf_matrix)
roc_val = plot_roc_curve(model_dectree,X_test,y_test) # plot the roc curve



decision_tree_plot = plot_model(model_dectree,['not churn','churn'])
plt.savefig("Decision_Tree_plot.png")




#permutation feature 


importance_df = plot_perm_importance(model_dectree, X_test, y_test)
print(importance_df)
plt.savefig("featureImportance.png")

X_train, X_test, y_train, y_test = prepare_model_smote(df, 'Churn', ['CustomerID'])
model, predictions, churn_list = execute_model(X_train, X_test, y_train, y_test)

pickle.dump(model_dectree, open('model.pkl', 'wb'))