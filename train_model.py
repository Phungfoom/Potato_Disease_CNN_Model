import os
import argparse
import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from tqdm import tqdm

import config
from Potato_bunch_brain import build_combined_model
from grad_cam_visualizer import visualize_gradcam_batch

# Focus on leaf and tube for foundation, field is midterm
parser = argparse.ArgumentParser(description="Train Potato ID Fusion Model")
parser.add_argument('--domain', type=str, required=True, choices=['leaf', 'tube', 'field'])
parser.add_argument('--stage', type=str, default="foundation", choices=['foundation', 'midterm'])
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
DOMAIN = args.domain
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Train on foundation, then midterm for fields, and NASA API for final check
if args.stage == "foundation":
    data_dir = os.path.join(script_dir, config.BASE_DIR, f'{DOMAIN}_classes')
else:
    # Midterm uses a different directory for "Field" tests
    data_dir = os.path.join(script_dir, config.BASE_DIR, f'midterm_{DOMAIN}_data')

train_raw = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="training",
    color_mode="rgb",
    shuffle=True,
    **config.DATA_PARAMS)

# Compute class weights to handle imbalanced datasets
y_train = np.concatenate([y for _, y in train_raw], axis=0)
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train)
class_weights_dict = dict(enumerate(weights))

class_names = train_raw.class_names
num_classes = len(class_names)

val_raw = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.3,
    subset="validation",
    color_mode="rgb",
    shuffle=True,
    **config.DATA_PARAMS)

val_labels = np.concatenate([y for _, y in val_raw], axis=0)
print("Validation images per class:", dict(zip(class_names, np.bincount(val_labels))))

# Prepare triple inputs (RGB, grayscale, Sobel) for the fusion model
def prepare_triple_input(image, label):
    """Convert RGB image to normalized triple: (RGB, grayscale, Sobel edge)."""
    norm_image = tf.cast(image, tf.float32) / 255.0
    gray_image = tf.image.rgb_to_grayscale(norm_image)
    sobel_image = tf.image.sobel_edges(gray_image)
    sobel_image = tf.reduce_sum(tf.abs(sobel_image), axis=-1)
    return (norm_image, gray_image, sobel_image), label

train_dataset = train_raw.map(prepare_triple_input).cache('spud_cache').prefetch(tf.data.AUTOTUNE)
val_dataset = val_raw.map(prepare_triple_input).cache('spud_cache').prefetch(tf.data.AUTOTUNE)

def plot_training_history(history, plot_dir, domain, stage, timestamp):
    """Plot training and validation accuracy/loss curves side by side."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy plot
    axes[0].plot(range(len(history.history['accuracy'])), 
                 history.history['accuracy'], label='Train Acc')
    axes[0].plot(range(len(history.history['val_accuracy'])), 
                 history.history['val_accuracy'], label='Val Acc')
    axes[0].set_title(f'Accuracy: {domain} ({stage})')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)

    # Loss plot
    axes[1].plot(range(len(history.history['loss'])), 
                 history.history['loss'], label='Train Loss')
    axes[1].plot(range(len(history.history['val_loss'])), 
                 history.history['val_loss'], label='Val Loss')
    axes[1].set_title(f'Loss: {domain} ({stage})')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'history_{domain}_{stage}_{timestamp}.png'))
    plt.close()

model = build_combined_model(num_classes=num_classes)

# Set up output directories
model_dir = os.path.join(script_dir, config.OUTPUT_DIR, 'fusion', 'models')
plot_dir = os.path.join(script_dir, config.OUTPUT_DIR, 'fusion', 'plots')
os.makedirs(model_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=config.EPOCHS,
    class_weight=class_weights_dict)

model.save(os.path.join(model_dir, f'potato_{DOMAIN}_model_{timestamp}.keras'))
plot_training_history(history, plot_dir, DOMAIN, args.stage, timestamp)

# Evaluate model on validation set
y_true, y_pred = [], []
for images, labels in tqdm(val_dataset, desc="Evaluating"):
    preds = model.predict_on_batch(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))


def save_report_as_image(y_true, y_pred, target_names, plot_dir, timestamp):
    """Save classification report as an image table."""
    # Get only classes that appear in the validation set
    present_classes = np.unique(y_true)
    filtered_names = [target_names[i] for i in present_classes]

    # Create the report as a dictionary and convert to DataFrame
    report = classification_report(
        y_true, y_pred, 
        labels=present_classes, 
        target_names=filtered_names, 
        output_dict=True)
    df = pd.DataFrame(report).transpose().round(2)

    plt.figure(figsize=(8, 5))
    plt.table(cellText=df.values, rowLabels=df.index, colLabels=df.columns, loc='center')
    plt.axis('off')
    plt.title(f"Results: {DOMAIN}")
    plt.savefig(os.path.join(plot_dir, f'report_{DOMAIN}_{timestamp}.png'), bbox_inches='tight')
    plt.close()


save_report_as_image(y_true, y_pred, class_names, plot_dir, timestamp)

# Generate confusion matrices (raw counts and percentages)
cm = confusion_matrix(y_true, y_pred)

# Calculate percentage confusion matrix
row_sums = cm.sum(axis=1)[:, np.newaxis]
cm_perc = np.divide(
    cm.astype('float'), row_sums,
    out=np.zeros_like(cm, dtype=float),
    where=row_sums != 0)

# Determine figure size based on number of classes
num_classes = len(class_names)
fig_size = max(10, num_classes * 0.8)

# Plot confusion matrix (percentages) - cleaner for reports
plt.figure(figsize=(fig_size, fig_size))
sns.heatmap(
    cm_perc * 100,  # Convert to percentages
    annot=True,
    fmt='.1f',
    cmap='Blues',
    cbar_kws={'label': 'Percentage (%)'},
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
    linecolor='gray')
plt.title(f'Confusion Matrix: {DOMAIN.upper()} Domain', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f'cm_{DOMAIN}_{timestamp}.png'), 
            bbox_inches='tight', dpi=300)
plt.close()

# Plot confusion matrix (raw counts) - for detailed analysis
plt.figure(figsize=(fig_size, fig_size))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='YlOrRd',
    cbar_kws={'label': 'Count'},
    xticklabels=class_names,
    yticklabels=class_names,
    linewidths=0.5,
    linecolor='gray')
plt.title(f'Confusion Matrix (Counts): {DOMAIN.upper()} Domain', fontsize=16, fontweight='bold', pad=20)
plt.ylabel('Actual Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(plot_dir, f'cm_counts_{DOMAIN}_{timestamp}.png'), 
            bbox_inches='tight', dpi=300)
plt.close()

# Generate Grad-CAM visualizations
visualize_gradcam_batch(
    model=model,
    validation_data=val_dataset,
    save_dir=plot_dir,
    name_tag=timestamp,
    class_names=class_names,
)