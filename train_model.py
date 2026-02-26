import os
import argparse
import tensorflow as tf
import config
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

from Potato_bunch_brain import build_combined_model
from sklearn.metrics import classification_report, confusion_matrix

# Focus on leaf and tubr for foundation, field is midterm
parser = argparse.ArgumentParser(description="Train Potato ID Fusion Model")
parser.add_argument('--domain', type=str, required=True, choices=['leaf', 'tube', 'field'])
parser.add_argument('--stage', type=str, default="foundation", choices=['foundation', 'midterm'])
args = parser.parse_args()

script_dir = os.path.dirname(os.path.abspath(__file__))
DOMAIN = args.domain

# Train on foundation, then midterm for fields, and NASA API for final check
if args.stage == "foundation":
    print(f"--- STAGE 1: {DOMAIN.upper()} Foundation Training ---")
    # Dynamically find leaf_classes or tube_classes
    data_dir = os.path.join(script_dir, config.BASE_DIR, f'{DOMAIN}_classes')
else:
    # Field domain for midterm
    data_dir = os.path.join(script_dir, config.BASE_DIR, f'{DOMAIN}_classes')


train_raw = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.3,
    subset="training",
    color_mode = "rgb",
    shuffle = True,
    **config.DATA_PARAMS)

val_raw = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split = 0.3,
    subset="validation",
    color_mode = "rgb",
    shuffle = False,
    **config.DATA_PARAMS)

class_names = train_raw.class_names
num_classes = len(class_names)

# Inputs for 3 branches 
def prepare_triple_input(image, label):
    norm_image = tf.cast(image, tf.float32) / 255.0 # Normalize 
    gray_image = tf.image.rgb_to_grayscale(norm_image)
    return (norm_image, gray_image, gray_image), label # rgb, gray, sobel 

train_spud = train_raw.map(prepare_triple_input).cache().prefetch(tf.data.AUTOTUNE)
val_spud = val_raw.map(prepare_triple_input).cache().prefetch(tf.data.AUTOTUNE)

# training history 
def plot_training_history(history, plot_dir, domain, stage):
    sns.set_theme(style = "whitegrid")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Accuracy
    sns.lineplot(x = range(len(history.history['accuracy'])), 
                 y = history.history['accuracy'], label = 'Train Acc', ax = ax1)
    sns.lineplot(x=range(len(history.history['val_accuracy'])), 
                 y = history.history['val_accuracy'], label = 'Val Acc', ax = ax1)
    ax1.set_title(f'Accuracy: {domain} ({stage})')

    # Loss
    sns.lineplot(x = range(len(history.history['loss'])), 
                 y = history.history['loss'], label = 'Train Loss', ax = ax2)
    sns.lineplot(x = range(len(history.history['val_loss'])), 
                 y = history.history['val_loss'], label = 'Val Loss', ax = ax2)
    ax2.set_title(f'Loss: {domain} ({stage})')

    plt.savefig(os.path.join(plot_dir, f'history_{domain}_{stage}.png'))
    plt.close()

model = build_combined_model(num_classes = num_classes)

# Folders for outputs
model_dir = os.path.join(script_dir, config.OUTPUT_DIR, 'fusion', 'models')
plot_dir = os.path.join(script_dir, config.OUTPUT_DIR, 'fusion', 'plots')
os.makedirs(model_dir, exist_ok = True)
os.makedirs(plot_dir, exist_ok = True)

history = model.fit(
    train_spud,
    validation_data = val_spud,
    epochs = config.EPOCHS)

model.save(os.path.join(model_dir, f'potato_{DOMAIN}_fusion_model.keras'))
plot_training_history(history, plot_dir, DOMAIN, args.stage)

# Confusion Matrtix for predictions
y_true = []
y_pred = []
total_batches = tf.data.experimental.cardinality(val_spud).numpy()

for images, labels in tqdm(val_spud, total = total_batches, desc = "Evaluating"):
    preds = model.predict(images, verbose = 0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis = 1))

# Classification Report
for images, labels in tqdm(val_spud, total = total_batches, desc = "Evaluating"):   
    preds = model.predict(images, verbose = 0)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis = 1))

print(classification_report(y_true, y_pred, target_names = class_names))


cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize = (10, 8))
sns.heatmap(cm, annot = True, fmt = 'd', 
            xticklabels = class_names, 
            yticklabels = class_names)
plt.title(f'Confusion Matrix: {DOMAIN} Fusion Model')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))