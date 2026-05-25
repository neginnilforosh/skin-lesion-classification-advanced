import itertools
import os
from glob import glob
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from PIL import Image

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical



def plot_model_history(model_history):
    """Plot training/validation accuracy, validation macro-F1 if available, and loss."""
    history = model_history.history
    acc_key = 'accuracy' if 'accuracy' in history else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history else 'val_acc'
    epochs_range = range(1, len(history[acc_key]) + 1)
    tick_step = max(1, len(history[acc_key]) // 10)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history[acc_key], label='train accuracy')
    plt.plot(epochs_range, history[val_acc_key], label='validation accuracy')
    if 'val_macro_f1' in history:
        plt.plot(epochs_range, history['val_macro_f1'], label='validation macro-F1')
    plt.title('Training Accuracy and Validation Macro-F1')
    plt.ylabel('Score')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(1, len(history[acc_key]) + 1, tick_step))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history['loss'], label='train loss')
    plt.plot(epochs_range, history['val_loss'], label='validation loss')
    plt.title('Training and Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.xticks(np.arange(1, len(history['loss']) + 1, tick_step))
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()


def extract_color_hist_features(images, bins=16):
    """Extract simple RGB histogram features from a list/array of images."""
    features = []
    for image in images:
        image = np.asarray(image).astype(np.uint8)
        image_features = []
        for channel in range(3):
            hist, _ = np.histogram(
                image[:, :, channel],
                bins=bins,
                range=(0, 256),
                density=True,
            )
            image_features.extend(hist)
        features.append(image_features)
    return np.asarray(features)


def evaluate_predictions(model_name, y_true, y_pred):
    """Return the main metrics used to compare all models."""
    return {
        'Model': model_name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced accuracy': balanced_accuracy_score(y_true, y_pred),
        'Macro F1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'Weighted F1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }


def make_metric_table(results):
    """Create a readable comparison table from a list of metric dictionaries."""
    metric_table = pd.DataFrame(results)
    numeric_cols = [col for col in metric_table.columns if col != 'Model']
    metric_table[numeric_cols] = metric_table[numeric_cols].round(4)
    return metric_table


def plot_labeled_confusion_matrix(cm, class_names, normalize=False, title='Confusion matrix'):
    """Plot a confusion matrix with explicit class labels."""
    cm_to_plot = cm.astype(float)

    if normalize:
        row_sums = cm_to_plot.sum(axis=1, keepdims=True)
        cm_to_plot = np.divide(
            cm_to_plot,
            row_sums,
            out=np.zeros_like(cm_to_plot),
            where=row_sums != 0,
        )
        fmt = '.2f'
    else:
        cm_to_plot = cm_to_plot.astype(int)
        fmt = 'd'

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_to_plot,
        annot=True,
        fmt=fmt,
        xticklabels=class_names,
        yticklabels=class_names,
        cmap='Blues',
        cbar=True,
    )
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()


def combine_training_histories(*histories):
    """Combine several Keras History objects so the training curves can be plotted together."""
    combined = {}
    for history_object in histories:
        for key, values in history_object.history.items():
            combined.setdefault(key, []).extend(values)
    return SimpleNamespace(history=combined)
