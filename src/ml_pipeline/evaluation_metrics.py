# function to create a confusion matrix
def evaluate_classification(true_labels, predicted_labels):
    """
    Calculate and display confusion matrix metrics
    """
    from sklearn.metrics import confusion_matrix
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(cm)
    
    # Extract values from confusion matrix
    true_negative, false_positive, false_negative, true_positive = cm.ravel()
    
    # Print individual metrics
    print(f'True Negatives: {true_negative}')
    print(f'True Positives: {true_positive}')
    print(f'False Positives: {false_positive}')
    print(f'False Negatives: {false_negative}')


def plot_roc_curve(model, features_test, labels_test):
    """
    Create and display ROC curve for classification model
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    
    # Calculate ROC metrics
    auc_score = roc_auc_score(labels_test, model.predict(features_test))
    false_positive_rate, true_positive_rate, _ = roc_curve(labels_test, model.predict(features_test))
    
    # Set up the plot
    plt.figure()
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    
    # Plot diagonal reference line (random classifier)
    plt.plot([0, 1], [0, 1], 'b--', label='Random Classifier')
    
    # Plot the model's ROC curve
    plt.plot(false_positive_rate, true_positive_rate, 
             color='darkorange', 
             label=f'Model (AUC = {auc_score:.2f})')
    
    # Add labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Save the plot
    plt.savefig('ROC_Curve.png')
  