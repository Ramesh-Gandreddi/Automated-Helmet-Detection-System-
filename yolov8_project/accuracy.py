import numpy as np

def calculate_metrics(true_labels, predictions):
    # Convert to numpy arrays for easier calculations
    true_labels = np.array(true_labels)
    predictions = np.array(predictions)

    # True Positives, False Positives, False Negatives
    TP = np.sum((true_labels == 1) & (predictions == 1))
    FP = np.sum((true_labels == 0) & (predictions == 1))
    FN = np.sum((true_labels == 1) & (predictions == 0))

    # Precision, Recall, F1 Score
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Accuracy
    accuracy = (TP + (len(true_labels) - TP - FP - FN)) / len(true_labels)

    return {
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1_score,
        'Accuracy': accuracy
    }

# Example usage
true_labels = [1, 0, 1, 1, 0]  # Ground truth (1 for helmet, 0 for no helmet)
predictions = [1, 0, 0, 1, 1]   # Model predictions

metrics = calculate_metrics(true_labels, predictions)

print("Metrics:")
print(f"Precision: {metrics['Precision']:.2f}")
print(f"Recall: {metrics['Recall']:.2f}")
print(f"F1 Score: {metrics['F1 Score']:.2f}")
print(f"Accuracy: {metrics['Accuracy']:.2f}")
