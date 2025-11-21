import pandas as pd


def plot_perm_importance(model, X_test, y_test):
    """Function to calculate and plot feature importance"""
    from sklearn.inspection import permutation_importance
    import matplotlib.pyplot as plt
    
    # Calculate importance
    result = permutation_importance(model, X_test, y_test, n_repeats=5)
    
    # Create dataframe
    df = pd.DataFrame({
        'feature': X_test.columns,
        'importance': result.importances_mean
    }).sort_values('importance', ascending=True)  # Sort for better plot
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(df['feature'], df['importance'])
    plt.xlabel('Feature Importance')
    plt.title('Permutation Feature Importance')
    plt.tight_layout()
   
   
    return df