import logging

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def display_metrics(y_true: np.ndarray, predicted: np.ndarray):
    labels = ['Ruy Lopez', 'Italian Game', 'Queen\'s Gambit', 'Sicilian Defense', 'Nimzo-Indian Defense']
    # labels = [f"Class {i}" for i in range(len(np.unique(y_true)))]

    logging.info('Generating classification report and confusion matrix.')
    print('Classification Report:\n', classification_report(y_true, predicted, target_names=labels, zero_division=0))
    print(f"Accuracy: {accuracy_score(y_true, predicted):.4f}")
    print(f"Precision: {precision_score(y_true, predicted, average='weighted'):.4f}")
    print(f"Recall: {recall_score(y_true, predicted, average='weighted'):.4f}")
    print(f"F1 Score: {f1_score(y_true, predicted, average='weighted'):.4f}")

    cm = confusion_matrix(y_true, predicted)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45)
    plt.title("Confusion Matrix")
    plt.show()
